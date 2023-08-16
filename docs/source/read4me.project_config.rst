
The project uses some configuration files,
a `python virtual environment <https://docs.python.org/3/library/venv.html#module-venv>`_, and some NLP models.

Follow these steps after having cloned or downloaded the project:

1. Customize the project .ini files
***********************************

The **config.ini** file contains parameter values and paths to the data and models used by the bot.
If you want to only create a dataset, you just need to download and extract a :ref:`word2vec <word2vec>` model
and change the "word2vec" entry to match its path.

The **telegram_token.ini** file contains the token for the bot.
To obtain a token, follow `this guide <https://core.telegram.org/bots/features#botfather>`_.

Both config.ini and telegram_token.ini need to reside in the root directory of the project.

2. Virtual environment (venv)
*****************************

Create and activate a `virtual environment <https://docs.python.org/3/library/venv.html#module-venv>`_ with at least **python 3.11** as interpreter.

.. code-block:: console

    user@userpc:~$ python3.11 -m venv venv
    user@userpc:~$ source venv/bin/activate

From the activated environment, use pip to install the requirements, listed in requirements.txt in the root
directory of the project.

.. code-block:: console

    (venv) user@userpc:~$ python -m pip install -r requirements.txt

.. warning::
    The python-telegram-bot library needs to be installed with the option [job-queue] in order for the
    :attr:`~telegram.ext.ConversationHandler.TIMEOUT` feature to work.
    When installing, make sure that the entry "python-telegram-bot[job-queue]" in requirements.txt is present,
    otherwise reinstall it manually later.

3. Install the project
**********************

From the activated environment, install the project as a library in it:
Install the project in the activated environment as a library

.. code-block:: console

    (venv) user@userpc:~$ python -m pip install -e .

The -e option ensures that the installation stays consistent with the changes you make to the project, without
reinstalling it every time.
You need to reinstall it only if you modify the pyproject.toml file.

4. Generate topics and models
*****************************

The bot uses a list of semantic topic vectors to look for similarity between the topics and the texts passed by the user.
It also needs a tfidf model paired with its dictionary.

| There are two scripts ready to use that generate all necessary models and data:
| See :ref:`here <make_corpus>` for how to make a corpus and generate the dictionary and the Tfidf model.
| See :ref:`here <topics_script>` for how to generate the topics.

If you want, you can plug in your own :class:`~gensim.corpora.dictionary.Dictionary` and :class:`~gensim.models.tfidfmodel.TfidfModel`.
See :ref:`Dictionary <dictionary>` and :ref:`Tfidf <tfidf>` for how to generate them.
If you use your models, then you need to update the appropriate paths in config.ini.

5. Configure the bot with BotFather
***********************************

The bot needs telegram inline mode to be active,
follow the `official telegram guide <https://core.telegram.org/bots/inline>`_ to activate it.

The inline query of the bot takes a web page link as input, therefore provide an appropriate placeholder for it during
the configuration with BothFather.

.. _run_bot:

6. Run the bot
**************

From the activated environment, use the command "run_bot" as stated in pyproject.toml in the section [project.scripts];
use ctrl+c to stop it.

.. code-block:: console

    (venv) user@userpc:~$ run_bot

The bot data file is generated and updated at runtime, its path is defined by the entry "telegram_pickle_persistence"
in the config.ini file.

.. note::
    The first time the bot is run, it might be slow to respond to a user request, this is caused by the library
    `fasttext-langdetect <https://pypi.org/project/fasttext-langdetect/>`_ that needs to download
    its language detection model before using it on texts.
