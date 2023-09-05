Read4Me bot user guide
======================

The main goal of the bot is to let you save some time online by letting it read the articles for you.

It may have happened to you to look for information online where you had to open and read several long pages,
many of which were either not what you were looking for, or did not treat some aspects deeply enough to be useful for you.
You still had to spend the time reading them nevertheless.

Here the bot can help by reading the content of the article for you and extracting relevant pieces from it.

If you send the bot a link to an article, the bot will show you some excerpts from it, each related to one
of the main subjects the article talks about.

The bot can also learn which kind of articles you usually like to read, you only need to leave a feedback after you read one article;
the next time, along with the excerpts, the bot will give you a prediction of interest based on your preferences.
This should help you choose which articles are worth their reading time.

.. note::
    The bot uses a fixed list of topics when it analyses documents, these topics are generated from a
    `News Dataset <https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html>`_
    that may lack some topics not generally treated in the news.
    In this case, the results and predictions may not be good enough for your use case.
    The topics can be updated while keeping what the bot have learned.

You can also put the focus of the bot on specific topics that you can manually define,
the bot will show them to you when you provide an article that treats them.

You can interact with it in three ways:

#. Obtain an interest prediction, based on learned personal preferences.
#. Manage your custom reading topics.
#. Obtain selected web page excerpts using telegram's inline mode feature.

1. Obtain interest prediction
*****************************

You can get information on a web page by sending a link to the bot's chat.
This will show you a prediction, based on your reading interests.
At the end, the bot will ask you to leave a feedback for your interest in the article.

.. mermaid:: requestlinkinfo.mmd

.. note::
    The bot learns your reading preferences, but it does not collect any information that can
    be used to log your reading history nor any other data, not even the feedback score you left.
    It only saves your telegram id.

The bot has internally a series of topics, shared among all users, that it uses when processing a document.
Whenever you leave a feedback, the bot updates your reading preferences by using the computed mixture of
topics found in the web page along with your feedback value.


2. Manage your custom reading topics
************************************

You can define your custom topics, they will be used to analyse the content of the web pages you provide.

.. note::

    In the bot's context, a topic is just a word cloud, such as "money investment stock finance" or "festival music dance".


In the bot's chat, type the command **/topics**.



You will be presented with the topics that you have and with up to four choices:

* Add a topic
* Delete a topic
* Delete all topics
* Exit

.. mermaid:: managecustomtopics.mmd

.. note::

    * Typing **help** from anywhere will let you see a guide for the custom topics.
    * Typing **stop** from anywhere will close the conversation.

You can add as many topics as you like, you can decide to use them to identify web content that you like or that you don't.
You can choose to delete your topics one by one or all of them at once.
After every change, the bot will show you the updated list of custom topics.


3. Obtain web page excerpts
***************************

You can provide the bot a link to a web page from every chat, using the `inline telegram mode <https://core.telegram.org/bots/inline>`_.

| **Use a space between the bot's name and the link you provide!**

.. mermaid:: inlinerequestlinkinfo.mmd

Bot execution
*************

To run the bot, see :ref:`here <run_bot>`.

Bot source code
***************

Press "source" to see the code or explore the full project on github.

.. automodule:: read4me.telegram_module
   :members:
