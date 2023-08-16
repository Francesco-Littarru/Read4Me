"""
Automatic download and extraction of warc archives and corpus creation from the CommonCrawl News Dataset.

The following options are available:

* **index** - File with the list of partial warc urls to download, specific to cc-news for this project.
* **first** - First url to process, starting from 0 as the first line of the index.
* **last** - Last url to process.
* **check_local_archive** (DEFAULT True) - Use local warc archive if available, download only if not already present.
* **keep_archive** (DEFAULT True) - Do not delete warc archive after download.
* **check_txt** (DEFAULT True) - Do not extract the archive if its relative text file is already present.
* **process_corpus** (DEFAULT False) - Perform processing of every available txt file in a separate pickle file.
* **process_missing** (DEFAULT False)- limit processing only to pickle corpus files not already processed.

Some parameter have a negative counterpart, use --help to get all the options and command abbreviations.

.. note::
    Archive extraction takes most of the time.

The following entries in the config.ini file of the project need to be set:

* **corpus_dir**
* **index_file**
* **status_file**
* **min_doc_len**
* **vocabulary_txt**
* **dictionary**
* **bow_id_corpus**
* **tfidf**

* **n_process**

Usage examples:

Download and extract the first three archives of the index, do not download those already downloaded,
delete the archives after extraction.

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.make_corpus 0 2 --check_local_archive --no-keep_archive

Download and extract the first three archives of the index, do not download those already downloaded or those for which
a text file exists (already extracted), delete the archives after extraction.

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.make_corpus 0 2 --check_local_archive --check_txt --no-keep_archive

Process the text files and create the normalized corpus of pickle files, do not process files already processed.

.. code-block:: console

    (venv) user@userpc:~$ python -m read4me.scripts.make_corpus 0 2 --process_missing --process_corpus

"""

import argparse
import configparser
import logging
from pathlib import Path
import pickle

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from read4me.datafactory import DataPaths, CorpusBuilder


logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(funcName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("make_corpus")
logger.setLevel(logging.INFO)


def make_corpus():
    """
    make_corpus script
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    defaults = config['DEFAULT']

    corpus_dir = Path(defaults['corpus_dir'])
    index_file = Path(defaults['index_file'])
    status_file = Path(defaults['status_file'])
    min_len = int(defaults['min_doc_len'])
    vocabulary_txt = Path(defaults['vocabulary_txt'])
    gensim_dict = Path(defaults['dictionary'])
    bow_id_corpus_path = Path(defaults['bow_id_corpus'])
    tfidf_path = Path(defaults['tfidf'])
    n_process = int(defaults['n_process'])

    parser = argparse.ArgumentParser(prog="make_corpus",
                                     description="Download, extract and clean texts from warc archives")
    parser.add_argument('-i', '--index', type=Path, default=index_file,
                        help='path to warc index file')
    parser.add_argument('first', type=int, default=0,
                        help='first file to process from index')
    parser.add_argument('last', type=int, default=0,
                        help='last file to process from index')
    parser.add_argument('--check_local_archive', type=bool, default=True,
                        help='use local archives if available',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-k', '--keep_archive', type=bool, default=True,
                        help='do not delete archives after download',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-c', '--check_txt', type=bool, default=True,
                        help='do not extract texts if available, extract only missing ones',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-p', '--process_corpus', default=False,
                        help='process the corpus of text files to generate the normalized corpus',
                        action='store_true')
    parser.add_argument('-m', '--process_missing', default=False,
                        help='process only text files not already processed, to use in conjunction with -p',
                        action='store_true')
    args = parser.parse_args()

    logger.info(f"\n{args}")
    paths = DataPaths(warc_index=index_file, range_start=args.first, range_stop=args.last,
                      text_vocab=vocabulary_txt, corpus_dir=corpus_dir)
    cb = CorpusBuilder(data_paths=paths)
    cb.set_status_path(status_file)
    cb.generate_texts_from_index(check_local_archive=args.check_local_archive,
                                 keep_archive=args.keep_archive,
                                 check_txt=args.check_txt,
                                 min_len=min_len)
    if args.process_corpus:
        cb.load_and_clean_origin(only_missing=args.process_missing, n_process=n_process)
        logger.info("Generating dictionary for corpus.")
        dictionary = Dictionary(cb.corpus)
        dictionary.save_as_text(str(gensim_dict))
        logger.info("Converting corpus in BoW format..")
        bow_id_corpus = [dictionary.doc2bow(doc) for doc in cb.corpus]
        with bow_id_corpus_path.open('wb') as file:
            pickle.dump(bow_id_corpus, file)
        logger.info("Computing TfIdf model..")
        tfidf = TfidfModel(dictionary=dictionary)
        tfidf.save(str(tfidf_path))
        cb.save_status()


if __name__ == '__main__':
    make_corpus()
