"""
**Required models and data**
****************************

#. Spacy model.
#. Dictionary.
#. Tfidf model.
#. Trained word2vec vectors as compatible KeyedVector instance.
#. A list of processed topics.

.. _spacy_model:

**1. Spacy model**
******************
From the active virtual environment install the spacy `en_core_web_md model` as described in the `official documentation <https://spacy.io/usage/models/#download>`_
or as in this example:

.. code-block:: console

    (venv) user@userpc:~$ python -m spacy download en_core_web_md

.. _dictionary:

**2. Dictionary**
*****************

The dictionary maps words to integer ids.
To create a :class:`~gensim.corpora.dictionary.Dictionary` use a corpus of documents where each document is a list of \
words, then save it in text format for later use and for seeing it directly in a text editor.

Example:

.. code-block::

    corpus = [['first','document'], ['another','document'], ['the','last','one']]
    dct = Dictionary(documents=corpus)
    dct.save_as_text('path_to_dictionary.txt')

.. _tfidf:

**3. Tfidf model**
******************

The :class:`~gensim.models.tfidfmodel.TfidfModel` is used to assign an importance value to a word in a document
accounting for the frequency of the word in the document and in the entire corpus. \
The model can be directly generated from the dictionary and then saved in a file.

.. code-block::

    dct = Dictionary.load_from_text('path_to_dictionary.txt')
    tfidf = TfidfModel(dictionary=dct)
    tfidf.save('tfidf_model')

.. _word2vec:

**4. Word2Vec**
***************

You can find different trained models ready to download here: http://vectors.nlpl.eu/repository/.
This project requires an english model with normalized vectors, optionally trained on a lemmatized corpus.
Download and extract the zip file in a directory of you choice, or use the models directory.
This project has been tested with the model with ID 11.

**5. Topics**
*************

For this project, use the script read4me/scripts/topics.py to generate the topics.
For more information about the script, in the activated virtual environment, type:

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.topics -h

The script can train a topic model and process the generated topics with
:class:`~read4me.topicsprocessor.TopicsProcessor`, the topics are then vectorized using a word2vec model
(the same returned from :func:`get_models`) and saved in a file.
"""

import configparser
from pathlib import Path
import pickle

from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, TfidfModel
import numpy
import spacy


# noinspection PyDefaultArgument
def get_models(args: list[str] = ['all']) -> list:
    """
    Get the models and data as requested.

    String parameters:

    #. 'nlp' - Spacy language model, see `spacy models <https://spacy.io/usage/models/>`_.
    #. 'dct' - :class:`~gensim.corpora.dictionary.Dictionary` obtained from a corpus.
    #. 'tfidf' - :class:`~gensim.models.tfidfmodel.TfidfModel` obtained from the dictionary.
    #. 'w2v' - :class:`~gensim.models.keyedvectors.KeyedVectors` trained model.
    #. 'topics' - list of topic vectors and separately their string descriptions.

    :param args: List of strings.
    :return: Python list of models and data in this order: [nlp, dct, tfidf, w2v, topics, topics_descriptions]
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    defaults = config['DEFAULT']

    _dict_path = Path(defaults['dictionary'])
    _tfidf_model_path = Path(defaults['tfidf'])
    _topics_path = Path(defaults['topics_vectors'])
    _w2v_path = Path(defaults['word2vec'])
    _topics_string_path = Path(defaults['topics'])

    mod_dict = []

    if 'nlp' in args or 'all' in args:
        mod_dict.append(spacy.load("en_core_web_md", disable=['ner']))
    if 'dct' in args or 'all' in args:
        mod_dict.append(Dictionary.load_from_text(str(_dict_path)))
    if 'tfidf' in args or 'all' in args:
        mod_dict.append(TfidfModel.load(str(_tfidf_model_path)))
    if 'w2v' in args or 'all' in args:
        mod_dict.append(KeyedVectors.load_word2vec_format(str(_w2v_path), binary=True))
    if 'topics' in args or 'all' in args:
        with _topics_path.open('rb') as _file:
            _topics_dict: dict = pickle.load(_file)
            mod_dict.append(numpy.asarray([_topics_dict[_vec] for _vec in sorted(_topics_dict.keys())]))
        with _topics_string_path.open('rb') as _file:
            mod_dict.append(pickle.load(_file))

    return mod_dict


if __name__ == '__main__':
    pass
