.. _inst_sec:

Follow these steps after having cloned or downloaded the project:

1. Customize the project .ini files
***********************************

The **config.ini** file contains default parameter values and paths to the data and models used by the bot.
By default, **read4me/data/corpus** and **read4me/data/models** are the directories for the corpus and the models that
will be used accordingly to the entries "corpus_dir" and "models_dir".
You can change these entries to directories of your choice.

The **telegram_token.ini** file contains the token for the bot.
To obtain a token, follow `this guide <https://core.telegram.org/bots/features#botfather>`_.
Make a telegram_token.ini file reusing this template:

.. code-block::

    [DEFAULT]
    token: **********************************************

Insert the token in place of the asterisks.

.. note::
    Both **config.ini** and **telegram_token.ini** need to reside in the root directory of the project.

2. Virtual environment (venv)
*****************************
Install first these two python dependencies:

.. code-block:: console

    user@userpc:~$ sudo apt-get install python3.11-venv python3.11-dev

Create and activate a `virtual environment <https://docs.python.org/3/library/venv.html#module-venv>`_ with at least **python 3.11** as interpreter.
You may create the virtual environment in the same directory of the project.

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

At the end install the spacy language model; see the :ref:`Spacy model <spacy_model>` section.

3. Install the project
**********************

Install the project in the activated environment as a pip library:

.. code-block:: console

    (venv) user@userpc:~$ python -m pip install -e .

The -e option ensures that the installation stays consistent with the changes you make to the project, without
reinstalling it every time.
You need to reinstall it only if you modify the pyproject.toml file.

Then run the dir_setup script that will generate the directories defined for the corpus and the models in the config.ini
file.

.. code-block:: console

    (venv) user@userpc:~$ dir_setup

4. Generate topics and models
*****************************

The bot uses:

* A list of semantic topic vectors to look for similarity between the topics and the texts passed by the user.
* A tfidf model paired with its dictionary.

To create the topic vectors you need to **download a Word2Vec model.** See :ref:`this section <word2vec>` for how to do it.

To generate the necessary models and data use the provided scripts:

1. Use the :ref:`make_corpus script <make_corpus>` to make a corpus with dictionary and tfidf model, starting from an index file.
2. Use the :ref:`topics script <topics_script>` to generate the topics and topic vectors.

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

The bot data file is generated and updated at runtime, its path for local storing is defined by the entry
"telegram_pickle_persistence" in the config.ini file.

.. note::
    The first time the bot is run, it might be slow to respond to a user request, this is caused by the library
    `fasttext-langdetect <https://pypi.org/project/fasttext-langdetect/>`_ that needs to download
    its language detection model before using it on texts.
