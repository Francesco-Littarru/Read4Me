import configparser
from enum import Enum
from functools import partial
import logging
import re
from uuid import uuid4

import numpy
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    MessageEntity,
    Update
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    filters,
    InlineQueryHandler,
    MessageHandler,
    PicklePersistence,
)

from read4me.custom_filters import filter_deny, filter_confirm, filter_bot_inline_input, filter_topics_delete
from read4me.datafactory import CorpusBuilder
from read4me.models import get_models
from read4me.userbase import UserBase, UserPreferences
from read4me.docprocessing import Doc

logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(funcName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("telegram_module")
logger.setLevel(logging.INFO)


def run():
    """
    Run the telegram bot.
    """

    # local constants
    _USER_BASE_ = "user_base"
    _LAST_MSG_ID_ = "last_message_id"
    _TIMEOUT_ = 5 * 60  # minutes * seconds

    # get the necessary models
    [nlp, dct, tfidf, w2v, topics, topics_description] = [*get_models(['all'])]
    t_ids = [td[0] for td in topics_description]

    CHOICE, ADD, DELETE, DELETE_ALL, LOOP_ADD, USER_VOTE = range(6)

    def italic(text: str) -> str:
        """
        Wrap with italic tag.
        """
        return f"<i>{text}</i>"

    def bold(text: str) -> str:
        """
        Wrap with bold tag.
        """
        return f"<b>{text}</b>"

    def scorer(score: float) -> str:
        """
        Compose a message to send to the user.
        The content of the message depends on the bot score.

        :param score: Float user-interest score assigned from the bot to a :class:`~read4me.processing.Doc`.
        :return: Message to send to the user.
        """
        if -0.6 <= score < -0.1:
            msg = "I don't think you would be interested in it"
        elif score < -0.6:
            msg = "Not very interesting"
        elif 0.1 < score <= 0.6:
            msg = "I think you would be interested in it"
        elif score > 0.6:
            msg = "Very interesting"
        else:
            msg = "I'm unsure about this, let me know if you read it."
        q = int(abs(score * 10))
        dash = "".join(['-' for _ in range(q - 1)])
        empty = "".join([' ' for _ in range(10 - q - 2)])
        space = "".join([' ' for _ in range(9)])
        if score < 0:
            a_message = f"(-){empty}<{dash}| {space}(+)"
        elif score > 0:
            a_message = f"(-){space} |{dash}>{empty}(+)"
        else:
            a_message = f"(-){space}<|>{space}(+)"
        return f"My interest prediction score for you is: {score:2.2f}\n{a_message}\n{msg}"

    def clear_tmp_data(context: ContextTypes.DEFAULT_TYPE):
        """
        Clear the last msg id from the bot data.
        """
        if _LAST_MSG_ID_ in context.bot_data:
            del context.bot_data[_LAST_MSG_ID_]

    async def link_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Show information of a document to the user.
        If the user has custom topics, show which ones are similar to the given article.
        At the end ask for feedback to learn the user preferences.
        """
        user_id = update.effective_user.id
        send_message = partial(context.bot.send_message, update.effective_chat.id)

        # check data availability from link before proceeding further
        await send_message("I'm processing your link, one moment please..")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        ent: MessageEntity = update.message.entities[0]
        link = update.message.text[ent.offset: ent.offset + ent.length]
        text = CorpusBuilder.doc_from_link(link)
        if not text:
            await send_message("I'm sorry, I couldn't retrieve the article.")
            clear_tmp_data(context)
            return ConversationHandler.END
        doc = Doc(text).process(w2v, tfidf, dct, nlp, {"VERB", "NOUN"}, topics, t_ids, CorpusBuilder.clean_doc)
        if doc is None:
            await send_message("I'm sorry, there was a problem extracting the article.")
            clear_tmp_data(context)
            return ConversationHandler.END
        context.user_data['doc'] = doc

        user_base: UserBase = context.bot_data[_USER_BASE_]
        msg = ""

        # build message for custom topics if the user has any
        if user_base.user(user_id).has_custom_topics():
            custom_topics = numpy.asarray(user_base.user(user_id).custom_topics)
            doc_custom: Doc = Doc(text). \
                process(w2v, tfidf, dct, nlp, {"VERB", "NOUN"}, custom_topics, text_cleaner=CorpusBuilder.clean_doc)
            score_custom = user_base.score_custom_topics(doc_custom, user_id)
            if bool(score_custom) and not doc_custom.has_triggered_fallback:
                for t_id, description in score_custom:
                    msg += "• " + description + "\n "
                await send_message("It matches with one or more of your topics!\n" + bold(msg), parse_mode='HTML')
                topics_ids = list(doc_custom.topic_counts.keys())
                msg = ""
                excerpts = set(doc_custom.excerpt(t) for t in topics_ids)
                for excerpt in excerpts:
                    msg += "\t" + excerpt + "\n"

        # proceed on scoring with the shared topics
        score = user_base.predict_interest(doc, user_id)
        await send_message(scorer(score))

        # add message to show the topics
        topics_ids = list(doc.topic_counts.keys())
        top_msg = "It seems to be talking about these topics:\n"
        if doc.has_triggered_fallback:
            top_msg = "The article is poorly related with these topics:\n"
        top_desc = dict(topics_description)
        top_desc_doc = [" ".join([w for (w, _) in t_s]) for t_s in
                        [top_desc[idx] for idx in topics_ids
                         if idx in top_desc.keys()]]
        for _topic in top_desc_doc:
            top_msg += "• " + bold(_topic) + "\n"
        if top_desc_doc:
            await send_message(top_msg, parse_mode="HTML")

        # continue adding excerpts from shared topics found in the article
        excerpts = set(doc.excerpt(t) for t in topics_ids)
        for excerpt in excerpts:
            msg += "\n\t" + excerpt + "\n"
        await send_message("From the article:\n" + italic(msg), parse_mode="HTML")

        # build the feedback-keyboard and show it to the user
        inline_keyboard = [[InlineKeyboardButton("-2", callback_data=-2),
                            InlineKeyboardButton("-1", callback_data=-1),
                            InlineKeyboardButton("0 (skip)", callback_data=0),
                            InlineKeyboardButton("1", callback_data=1),
                            InlineKeyboardButton("2", callback_data=2)]]
        reply_markup = InlineKeyboardMarkup(inline_keyboard)
        msg = await update.message.reply_text(
            text="How much would you rate your interest in this article?",
            reply_markup=reply_markup)
        context.user_data[_LAST_MSG_ID_] = msg.id
        return USER_VOTE

    async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Read the user feedback and update the preferences accordingly.
        """
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(text=f"Your vote: {query.data}")
        context.bot_data[_USER_BASE_].update_user_preferences(
            update.effective_user.id,
            context.user_data['doc'],
            float(query.data))
        del context.user_data['doc']
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Thanks for the feedback!")
        clear_tmp_data(context)
        return ConversationHandler.END

    async def waiting_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Send a warning message if the user is trying to issue another request before leaving the feedback.
        """
        text = "Please leave a feedback for the article before issuing other requests or type 'stop'."
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=text,
                                       reply_to_message_id=context.user_data[_LAST_MSG_ID_])

    async def end_state(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Close the conversation.
        """
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Bye!")
        clear_tmp_data(context)
        return ConversationHandler.END

    async def time_expired(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Close the conversation after some amount of time if the user was inactive while the bot was waiting.
        """
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="You didn't respond for quite some time, "
                                            "I'm closing the conversation. Bye!",
                                       reply_to_message_id=context.user_data[_LAST_MSG_ID_])
        clear_tmp_data(context)
        return ConversationHandler.END

    async def intro(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Show an introduction to the bot.
        """
        send_message = partial(context.bot.send_message, update.effective_chat.id)
        msg1 = "Hi!\n" \
               "This is the helper page of the Read4Me telegram bot.\n\n" \
               "My job is to help you save some of your time online.\n" \
               "I can read web pages and analyze their content.\n" \
               "Send me a link and I will show you the most relevant pieces of it.\n\n" \
               "Leave a feedback after you read the original page so I can learn your reading preferences " \
               "and advise you better on the next ones you send me. " \
               "(I won't send anything of what I learn to anyone)"
        msg2 = "Also you can define your /topics and I will match them with the pages you send me." \
               "\n\nTo read this guide again type /help\n" \
               "To see what you can do with the bot type /commands"
        await send_message(msg1)
        await send_message(msg2)

    async def commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Show a list of commands and inputs that the bot can accept.
        """
        msg = f"Hi, these are the commands and inputs that you can send me\n\n" \
              "/help - see the help guide\n" \
              "/topics - open the menu to define personalized topics\n" \
              "/topicsinfo - see how to define and use topics\n" \
              "/commands - see this list\n\n" \
              "<b>Send a link in this chat</b>\nObtain a prediction of interest thanks to your feedbacks and get " \
              "some of the most relevant sentences from the article\n\n" \
              f"<b>Call me from any chat and give me a link</b>, " \
              f"like this:\n {context.bot.name} <i><u>link</u></i> \n" \
              "I will send you some of the most relevant passages from the article"
        await context.bot.send_message(update.effective_chat.id, msg, parse_mode='HTML')

    async def post_init(application: Application):
        """
        Sync the topics in the :class:`~read4me.userbase.UserBase` with the system topics.
        This allows to use different topics and reuse the

        This executes BEFORE starting polling from telegram servers and after application.run_polling() is called.
        """
        if _USER_BASE_ not in application.bot_data:
            application.bot_data[_USER_BASE_] = UserBase(topics, topics_description)
        application.bot_data[_USER_BASE_].set_topics(topics, topics_description)

    async def topics_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Explanation on how to define topics.
        """
        msg = "A topic is a concept described with a group of words.\n\n" \
              "To define a personal topic, you can type a single word or a list of linked words, " \
              'for example: <b>\n - "music", \n - "food recipe", \n - "money investment stock market"</b>.\n' \
              'If you use single words, avoid generic terms such as <b>"news"</b>, ' \
              "as they tend to relate to almost anything in the web.\n\n" \
              "Each time you will send me an link to an article, " \
              "I will tell you which ones of your topics do match with it\n\n" \
              "Tip: If you define topics about articles that you don't like " \
              "you can spot them faster!\n\n" \
              "If you requested this help guide while typing a topic, now you can continue typing it."
        await context.bot.send_message(update.effective_chat.id, msg, parse_mode="HTML")
        return None

    async def topics_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Present an :class:`InlineKeyboard` to perform actions on custom topics.
        """
        send_message = partial(context.bot.send_message, update.effective_chat.id)
        user: UserPreferences = context.bot_data[_USER_BASE_].user(update.effective_user.id)
        inline_keyboard = [[InlineKeyboardButton("Add topic", callback_data=2)],
                           [InlineKeyboardButton("Exit", callback_data=1)]]
        if user.has_custom_topics():
            topics_vis: str = "Your topics:\n"
            for t_id, topic in enumerate(user.custom_topics_descriptions):
                topics_vis += f"\t{bold(str(t_id))} - {italic(topic)}\n"
            await send_message(topics_vis, parse_mode="HTML")
            inline_keyboard[0].append(InlineKeyboardButton("Delete topic", callback_data=3))
            inline_keyboard[0].append(InlineKeyboardButton("Delete all", callback_data=4))
        else:
            await send_message("You don't have custom topics.")
        reply_markup = InlineKeyboardMarkup(inline_keyboard)

        msg = await update.message.reply_text(text="What would you like to do?", reply_markup=reply_markup)
        context.user_data[_LAST_MSG_ID_] = msg.id
        return CHOICE

    async def choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Read the choice made by the user on the custom-topic :class:`InlineKeyboard`.
        """
        send_message = partial(context.bot.send_message, update.effective_chat.id)

        query = update.callback_query
        await query.answer()
        selected = int(query.data)
        selection = Enum('choice', ['Exit', 'Add topic', 'Delete one', 'Delete all'])
        await query.edit_message_text(text=f"You chose: {bold(selection(selected).name)}", parse_mode='HTML')
        if selected == 1:
            await send_message("Bye!")
            clear_tmp_data(context)
            return ConversationHandler.END
        elif selected == 2:
            msg = await context.bot.send_message(update.effective_chat.id,
                                                 "Type your topic, type 'help' if needed or 'stop' to quit.")
            context.user_data[_LAST_MSG_ID_] = msg.id
            return ADD
        elif selected == 3:
            msg = await send_message("Ok, give me the number of the topic to delete")
            context.user_data[_LAST_MSG_ID_] = msg.id
            return DELETE
        elif selected == 4:
            msg = await send_message("All your personal topics will be deleted:\n"
                                     "Do you confirm? (yes/no)")
            context.user_data[_LAST_MSG_ID_] = msg.id
            return DELETE_ALL

    async def topic_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Manage the insertion of a new user custom topic.
        """
        topic_text = update.message.text
        words = {w for w in nlp(re.sub(r"[^A_Za-z0-9]", " ", topic_text.lower())) if w.text not in {" ", ""}}
        oov = {w.text for w in words if w.lemma_ not in w2v.key_to_index}
        vecs = [w2v[w.lemma_] for w in words if w.text not in oov]
        if vecs:
            user: UserPreferences = context.bot_data[_USER_BASE_].user(update.effective_user.id)
            vec = numpy.sum(vecs, axis=0)
            vec = vec / numpy.linalg.norm(vec)
            if user.has_this_custom_topic(vec):
                await context.bot.send_message(update.effective_chat.id, "You have a very similar topic already saved!")
            else:
                if oov:
                    await context.bot.send_message(update.effective_chat.id,
                                                   f"I don't know these words and I didn't use them: {[*oov]}")
                user.add_custom_topic(vec, " ".join([w.text for w in words if w.lemma_ not in oov]))
                await context.bot.send_message(update.effective_chat.id, "Your topic has been saved!")
        else:
            if oov:
                await context.bot.send_message(update.effective_chat.id,
                                               f"I don't know any of these words: {[*oov]}")
                msg = await context.bot.send_message(update.effective_chat.id, f"Please try with other ones")
                context.user_data[_LAST_MSG_ID_] = msg.id
                return ADD
        msg = await context.bot.send_message(update.effective_chat.id,
                                             "Do you want to insert another topic? (yes/no)\ntype 'stop' to quit")
        context.user_data[_LAST_MSG_ID_] = msg.id
        return LOOP_ADD

    async def topic_add_again(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Ask the user to insert a topic.
        """
        msg = await context.bot.send_message(update.effective_chat.id, "Type your topic.")
        context.user_data[_LAST_MSG_ID_] = msg.id
        return ADD

    async def topics_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Manage the removal of a user custom topic.
        """
        topic_id = int(update.message.text)
        user: UserPreferences = context.bot_data[_USER_BASE_].user(update.effective_user.id)

        if 0 <= topic_id <= user.num_custom_topics - 1:
            user.delete_custom_topic(topic_id)
        else:
            await context.bot.send_message(
                update.effective_chat.id,
                f"There isn't a topic with number {topic_id}\nWhich topic do you want to delete?")
            if user.num_custom_topics > 1:
                await context.bot.send_message(update.effective_chat.id,
                                               f"You can choose a topic from 0 to {user.num_custom_topics - 1}")
            else:
                await context.bot.send_message(update.effective_chat.id,
                                               "Type 0 if you want to delete the last topic.")
            return DELETE
        msg = ""
        if user.has_custom_topics():
            topics_vis: str = "Your topics now:\n"
            for t_id, topic in enumerate(user.custom_topics_descriptions):
                topics_vis += f"\t{bold(str(t_id))} - {italic(topic)}\n"
            await context.bot.send_message(update.effective_chat.id, topics_vis, parse_mode="HTML")
            msg += "To delete another topic, type its number\n"
        else:
            await context.bot.send_message(update.effective_chat.id, "Your topic has been deleted\n")
            if not user.has_custom_topics():
                inline_keyboard = [[InlineKeyboardButton("Add topic", callback_data=2)],
                                   [InlineKeyboardButton("Exit", callback_data=1)]]
                reply_markup = InlineKeyboardMarkup(inline_keyboard)
                msg = await update.message.reply_text(text="What would you like to do?", reply_markup=reply_markup)
                context.user_data[_LAST_MSG_ID_] = msg.id
                return CHOICE
        msg += "Send /topics to go back or type 'stop' to quit."
        message = await context.bot.send_message(update.effective_chat.id, msg)
        context.user_data[_LAST_MSG_ID_] = message.id
        return DELETE

    async def topics_delete_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Delete all user custom topics.
        """
        user: UserPreferences = context.bot_data[_USER_BASE_].user(update.effective_user.id)
        user.delete_all_custom_topics()
        await context.bot.send_message(update.effective_chat.id,
                                       "All your topics have been deleted.\n"
                                       "Send /topics to show the menu or type 'stop' to quit.")
        return DELETE_ALL

    async def topic_not_found(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Manage an invalid input when the user should enter a custom topic as a list of words.
        """
        await context.bot.send_message(update.effective_chat.id,
                                       f"Sorry, but this doesn't seem to be a valid input,\n"
                                       f"please type the topic as a list of words.\n"
                                       f"Type 'help' if needed or 'stop' to quit.",
                                       reply_to_message_id=context.user_data[_LAST_MSG_ID_])
        return None

    async def wrong_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generic wrong input message.
        """
        await context.bot.send_message(update.effective_chat.id,
                                       "Please make a valid choice before continuing, or type 'stop' to quit.",
                                       reply_to_message_id=context.user_data[_LAST_MSG_ID_])
        return None

    async def unanswered(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Manage unanswered bot requests when the user tries to issue a new one.
        """
        await context.bot.send_message(update.effective_chat.id, "Please answer the question, or type 'stop' to quit.",
                                       reply_to_message_id=context.user_data[_LAST_MSG_ID_])
        return None

    async def inline_query(update, context):
        """
        Extract some excerpts from the document when the user performs an inline query.
        """
        link = update.inline_query.query
        if not link:
            return None

        def extract():
            text = CorpusBuilder.doc_from_link(link)
            if not text:
                return "I can't read the article, sorry"
            doc: Doc = Doc(text).process(w2v, tfidf, dct, nlp, {"VERB", "NOUN"}, topics, t_ids, CorpusBuilder.clean_doc)
            if not doc:
                return "I can't read the link, sorry"
            topics_ids = list(doc.topic_counts.keys())
            excerpts = set(doc.excerpt(t) for t in topics_ids)
            msg = ""
            for excerpt in excerpts:
                msg += "\n\t" + excerpt + "\n"
            return msg

        res = [
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="Get Article Info",
                input_message_content=InputTextMessageContent(message_text=f"{link}\n" + italic(extract()),
                                                              parse_mode='HTML')
            )
        ]
        await update.inline_query.answer(res)

    config = configparser.ConfigParser()
    config.read('telegram_token.ini')
    token = config['DEFAULT']['token']
    config.read('config.ini')
    pickle_tg_data = config['DEFAULT']['telegram_pickle_persistence']

    pickle_persistence = PicklePersistence(filepath=pickle_tg_data)
    application = ApplicationBuilder().token(token) \
        .persistence(pickle_persistence) \
        .post_init(post_init) \
        .build()

    application.add_handler(CommandHandler('start', intro))
    application.add_handler(CommandHandler('commands', commands))
    application.add_handler(CommandHandler('help', intro))
    application.add_handler(CommandHandler('topicsinfo', topics_help))
    application.add_handler(InlineQueryHandler(inline_query))

    application.add_handler(
        ConversationHandler(
            entry_points=[CommandHandler("topics", topics_request),
                          MessageHandler(~filter_bot_inline_input &
                                         filters.Entity(MessageEntity.URL), link_request)],
            states={
                CHOICE: [CallbackQueryHandler(choice),
                         MessageHandler(filters.TEXT & filters.Regex("^[Ss]top$"), end_state),
                         MessageHandler(filters.TEXT & filters.Regex("^[Hh]elp$"), topics_help),
                         MessageHandler(filters.ALL, wrong_input)],

                ADD: [MessageHandler(filters.TEXT & filters.Regex("^[Ss]top$"), end_state),
                      MessageHandler(filters.TEXT & filters.Regex("^[Hh]elp$"), topics_help),
                      MessageHandler(filters.TEXT & ~filters.COMMAND &
                                     ~(filter_confirm | filter_deny) &
                                     ~filters.Entity(MessageEntity.URL), topic_add),
                      MessageHandler(filters.ALL, topic_not_found)],

                LOOP_ADD: [MessageHandler(filters.TEXT & ~filters.COMMAND & filter_confirm, topic_add_again),
                           MessageHandler(filters.TEXT & ~filters.COMMAND & filter_deny, topics_request),
                           MessageHandler(filters.TEXT & filters.Regex("^[Hh]elp$"), topics_help),
                           MessageHandler(filters.TEXT & filters.Regex("^[Ss]top$"), end_state),
                           MessageHandler(filters.ALL, wrong_input)],

                DELETE: [MessageHandler(filters.TEXT & filter_topics_delete, topics_delete),
                         CommandHandler("topics", topics_request),
                         MessageHandler(filters.TEXT & filters.Regex("^[Ss]top$"), end_state),
                         MessageHandler(filters.TEXT & filters.Regex("^[Hh]elp$"), topics_help),
                         MessageHandler(filters.ALL, wrong_input)],

                DELETE_ALL: [MessageHandler(filters.TEXT & filter_confirm, topics_delete_all),
                             MessageHandler(filters.TEXT & filter_deny, topics_request),
                             MessageHandler(filters.TEXT & filters.Regex("^[Ss]top$"), end_state),
                             MessageHandler(filters.TEXT & filters.Regex("^[Hh]elp$"), topics_help),
                             CommandHandler("topics", topics_request),
                             MessageHandler(filters.ALL, unanswered)],

                USER_VOTE: [
                    CallbackQueryHandler(feedback),
                    MessageHandler(filters.TEXT & filters.Regex("^[Ss]top"), end_state),
                    MessageHandler(filters.ALL, waiting_score)],

                ConversationHandler.TIMEOUT: [MessageHandler(filters.ALL, time_expired)],
            },
            fallbacks=[],
            per_chat=False,
            conversation_timeout=_TIMEOUT_
        )
    )

    application.add_handler(MessageHandler(~filter_bot_inline_input, commands))
    application.run_polling()


if __name__ == '__main__':
    run()
