"""
Custom :mod:`~telegram.ext.filters` for the handlers inside :class:`~telegram.ext.ConversationHandler`
in :mod:`~read4me.telegram_module`.
These filters are needed to guide the conversation with the user and interpret the inputs.
"""

import re

from telegram.ext.filters import *


class FilterBotInlineInput(MessageFilter):
    """
    Respond only to user messages, ignore the message if the bot itself sent it as response to
    an inline query of the user in the bot's own chat.
    """
    def filter(self, message):
        return message.via_bot and message.via_bot.is_bot


class FilterTopicsDelete(MessageFilter):
    """
    The bot expects a number of a target topic to delete during the conversation with the user,
    no other input is allowed.
    """
    def filter(self, message):
        return re.fullmatch(r"[0-9]+", message.text) is not None


class FilterConfirm(MessageFilter):
    """
    The bot expects a "yes/no" answer. "yes" is received.
    """
    def filter(self, message):
        return message.text.lower() == "yes"


class FilterDeny(MessageFilter):
    """
    The bot expects a "yes/no" answer. "no" is received.
    """
    def filter(self, message):
        return message.text.lower() == "no"


filter_bot_inline_input = FilterBotInlineInput()
filter_topics_delete = FilterTopicsDelete()
filter_confirm = FilterConfirm()
filter_deny = FilterDeny()

if __name__ == '__main__':
    pass
