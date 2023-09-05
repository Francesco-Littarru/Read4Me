"""
Script for the training of the :mod:`~gensim.models.hdpmodel`, and for processing and vectorization of topics.

.. tip::

    Use this command to generate the topics if you are in a hurry!
    From the activated environment:

    .. code-block:: console

        (venv) user@userpc:~$ python -m read4me.scripts.topics --train --process_topics --show_saved --vectorize

    If you see errors during filtering or if you are not satisfied with the topics,
    try tuning some topic parameters such as min_dict_count in the config.ini file.
    
    Continue reading for more detailed information.

The following options are available:

* **train** (DEFAULT False) - Train the topic model and save the topic descriptions.
* **passes** (DEFAULT 1) - Train passes on the corpus.
* **process_topics** (DEFAULT False) - Process the topics applying filtering.
* **show_origin** (DEFAULT False) - Show all the topics extracted by the topic model.
* **show_saved** (DEFAULT False) - Show the topics obtained after filtering.
* **vectorize** (DEFAULT False) - Save the filtered topic as numpy vectors, using the word2vec model.

Use --help to get all the options and command abbreviations.

The following entries in the config.ini file of the project need to be set:

* **corpus_dir**
* **status_file**
* **dictionary**
* **bow_id_corpus**

* **topic_model**
* **topics**
* **topics_vectors**
* **word2vec**

* **alpha**
* **gamma**
* **num_topics**
* **num_words**
* **min_proba**
* **min_dict_count**
* **topic_coherence**

* **n_process**

Usage examples:

Train the topic model with 2 passes on the corpus and show the topics afterwards:

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.topics --train --passes 2 --show_origin

Process the topics and show the filtered result:

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.topics --process_topics --show_saved

Vectorize the filtered topics:

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.topics --vectorize

"""

import pickle
import numpy
import argparse
import configparser
import logging
from pathlib import Path
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import HdpModel, KeyedVectors
from numpy.typing import NDArray

from read4me.datafactory import CorpusBuilder
from read4me.models import get_models
from read4me.topicsprocessor import TopicsProcessor

logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(funcName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("make_topics")
logger.setLevel(logging.INFO)


def topics():
    """
    topics script
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    defaults = config['DEFAULT']

    status_file = Path(defaults['status_file'])
    topic_model = Path(defaults['topic_model'])
    topics_path = Path(defaults['topics'])
    alpha = int(defaults['alpha'])
    gamma = int(defaults['gamma'])
    num_topics = int(defaults['num_topics'])
    num_words = int(defaults['num_words'])
    min_proba = float(defaults['min_proba'])
    min_dict_count = float(defaults['min_dict_count'])
    topic_coherence = float(defaults['topic_coherence'])
    bow_id_corpus_path = Path(defaults['bow_id_corpus'])
    vectors = Path(defaults['topics_vectors'])
    n_process = int(defaults['n_process'])

    parser = argparse.ArgumentParser(prog="make_topics",
                                     description="Generate topics from cleaned corpus")
    parser.add_argument('-t', '--train', default=False,
                        help='train the topic model',
                        action='store_true')
    parser.add_argument('--passes', type=int, default=1,
                        help='training passes on the corpus')
    parser.add_argument('-p', '--process_topics', default=False,
                        help='process the topics applying filtering',
                        action='store_true')
    parser.add_argument('-o', '--show_origin', default=False,
                        help='show all original topics, ignore other settings',
                        action='store_true')
    parser.add_argument('-s', '--show_saved', default=False,
                        help='show saved topics, ignore other settings',
                        action='store_true')
    parser.add_argument('-v', '--vectorize', default=False,
                        help='create word2vec vectors for the topics',
                        action='store_true')

    args = parser.parse_args()

    def process_topics():
        cb = CorpusBuilder.load_status(status_file)
        cb.load_corpus()
        logger.info("Processing topics.")
        model_topics = hdp.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        tp = TopicsProcessor(
            topics=model_topics, corpus=cb.corpus, dictionary=dictionary,
            min_dict_count=min_dict_count, min_topic_proba=min_proba, min_topic_coherence=topic_coherence,
            n_process=n_process)
        tp.filter_topics()
        tp.save_topics(topics_path)

    if args.train:
        with bow_id_corpus_path.open('rb') as file:
            bow_id_corpus = pickle.load(file)
        dictionary: Dictionary = get_models(['dct'])[0]  # Dictionary.load_from_text(str(gensim_dict))
        logger.info(f"Training HdpModel.")
        max_chunks = int(len(bow_id_corpus)/256) * args.passes
        hdp = HdpModel(bow_id_corpus, dictionary, max_chunks=max_chunks, alpha=alpha, gamma=gamma)
        logger.info(f"Done. Saving to {topic_model}")
        hdp.save(str(topic_model))
        if args.process_topics:
            process_topics()
    elif args.process_topics:
        hdp = HdpModel.load(str(topic_model))
        dictionary: Dictionary = get_models(['dct'])[0]  # Dictionary.load_from_text(str(gensim_dict))
        process_topics()

    if args.show_saved:
        if not topics_path.is_file():
            logger.error(f"File not found {topics_path}. Did you process the topics?")
        else:
            logger.info("Loading saved topics.")
            with topics_path.open('rb') as file:
                saved_t = pickle.load(file)
            pprint(saved_t)

    if args.show_origin:
        if not topic_model.is_file():
            logger.error(f"File not found {topic_model}")
        else:
            logger.info("Loading topic model.")
            hdp = HdpModel.load(str(topic_model))
            pprint(hdp.show_topics(num_topics=150, formatted=False))

    if args.vectorize:
        if not topics_path.is_file():
            logger.error(f"File not found {topics_path}")
        else:
            with topics_path.open('rb') as file:
                topics = pickle.load(file)
            w2v: KeyedVectors = get_models(['w2v'])[0]
            topic_vectors: dict[int, NDArray] = {}
            if len(topics) > 0:
                for (t_id, topic) in topics:
                    words, weights = zip(*topic)
                    topic_vector = numpy.sum([w2v[word]*weights[i]
                                              for i, word in enumerate(words) if word in w2v.key_to_index],
                                             axis=0)
                    if topic_vector.__class__ == numpy.ndarray:
                        topic_vectors[t_id] = topic_vector/numpy.linalg.norm(topic_vector)
                with vectors.open('wb') as file:
                    pickle.dump(topic_vectors, file)
                    logger.info(f"Topic vectors saved to {vectors}")
            else:
                logger.error("There are no topics!")


if __name__ == '__main__':
    topics()
