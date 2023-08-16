Welcome to Read4Me's documentation!
===================================

Go the the :ref:`Main Telegram Bot <main_tg_bot>` section to see what you can do with the bot and how it works.

In the :ref:`Modules Documentation <modules_doc>` is described each module in detail with examples of usage.

In the :ref:`Scripts <scripts>` section you will find a brief tutorial on how to generate a news dataset
and how to create topic vectors for the bot.

To try the project or to modify it, see the :ref:`Installation <inst_sec>` section.

The project consists of:

- A **Telegram bot** that analyses the content of web pages.
- **Modules** to build a news dataset, to process topics and documents, and to manage telegram user's data.
- **Scripts** to automate the corpus and topics creation and processing.

A streamlined view of the full execution can be summarized in few steps:

#. The news dataset is built, and a list of topics is extracted from them.
#. The topics are further processed obtaining fewer and more interpretable ones.
#. The processed topics are then used by the bot as a base reference when analyzing documents and learning user
   reading preferences.


.. _main_tg_bot:
.. toctree::
    :caption: Telegram Bot
    :maxdepth: 1

    Main Telegram Bot <read4me.telegram_module>
    Telegram Custom Filters <read4me.custom_filters>

.. _inst_sec:
.. toctree::
    :caption: Installation
    :maxdepth: 1

    read4me.project_config

.. _modules_doc:
.. toctree::
    :caption: Modules Documentation
    :maxdepth: 1

    Dataset Creation module <read4me.datafactory>
    Process Topics <read4me.topicsprocessor>
    Process Documents <read4me.docprocessing>
    Manage users <read4me.userbase>
    Retrieve models <read4me.models>

.. _scripts:
.. toctree::
    :caption: Scripts
    :maxdepth: 1

    Corpus script <read4me.scripts.make_corpus>
    Topics script <read4me.scripts.topics>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
