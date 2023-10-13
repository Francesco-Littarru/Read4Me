"""
**Preliminary information.**
****************************

With the datafactory module you can build a corpus of text documents, using the News
Dataset repository of the open `CommonCrawl Project <https://commoncrawl.org/>`_.

**What classes are there?**

* :class:`DataPaths`: A metadata container used to initialize a CorpusBuilder instance.
* :class:`CorpusBuilder`: The corpus generator.

**How can you use it?**

.. note::

    To start building a corpus you need to download and extract one index file from
    the list of monthly News Dataset indexes at https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html.
    Select a year and a month from the table at the link previously provided, then extract the warc.paths file.
    An index file contains a list of partial URLs for a month worth of news articles.
    Initializing a DataPaths instance with the index will also complete the URLs with a valid prefix.

The extracted dataset files and the corpus files are stored respectively as text and pickle files
inside a dedicated directory of your choice.
The text files contain the original documents as extracted from the archives, one for each line, without duplicates,
while the pickle files contain the processed texts in binary format for easy memory loading.

Building and loading the corpus can be done in few steps:

1. Define the metadata.
2. Generate the corpus.
3. Load and use the corpus.

**1. Define the metadata.**
***************************

Initialize a DataPaths instance with the index file and other information.

Example:
Use an index file called "warc.paths", select from it only the first
10 lines (archives) to be used.
The designated path to save the corpus files is the directory
"corpus/" and the vocabulary will be saved inside of it as "vocab.txt".
If the directory "corpus/" is not already present it will be created.

.. code-block::

    datapaths = DataPaths(
        warc_index=pathlib.Path("warc.paths"),
        range_start=0, range_stop=9,
        text_vocab=pathlib.Path("corpus/vocab.txt"),
        corpus_dir=pathlib.Path("corpus"))``

The metadata is ready.

**2. Generate the corpus.**
***************************

You can generate the corpus in few steps:

Initialize an instance with the metadata and then set where to save the status file.
The status file contains the metadata and other information to reload the corpus once it has been generated.

.. code-block::

    cb = CorpusBuilder(datapaths)
    cb.set_status_path(pathlib.Path("status.pkl"))

Download and extract the archives into txt files.
The archives are those selected from the index file.

.. code-block::

    cb.generate_texts_from_index()

Process the text files and generate the corpus pickle files.

.. code-block::

    cb.load_and_clean_origin()

Save the status of the corpus into a file.
The status contains all relevant information for loading the corpus.
The corpus is not serialized because it is already stored separately in the pickle files.

.. code-block::

    cb.save_status()

**3. Load and use the corpus.**
*******************************

First load the status containing the metadata, then load the corpus, using the metadata to locate the right corpus files.

.. code-block::

    cb = CorpusBuilder.load_status()
    cb.load_corpus()

The :attr:`corpus <CorpusBuilder.corpus>` is formatted as a list of documents,
where each document can be either a string or a list of words, the default is a list of words.
"""

from collections import defaultdict
from functools import partial
import logging
from pathlib import Path
import pickle
import re
import subprocess
from typing import Self, Iterator

from ftlangdetect import detect
import spacy
import trafilatura
from warcio.archiveiterator import ArchiveIterator
import wget

tlog = logging.getLogger("trafilatura")
tlog.propagate = False  # deactivate logging from trafilatura
logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(funcName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("datafactory")
logger.setLevel(logging.INFO)


class DataPaths:
    """
    Setup of the metadata for the creation of the corpus.
    """

    def __init__(self,
                 warc_index: Path,
                 range_start: int,
                 range_stop: int,
                 text_vocab: Path,
                 corpus_dir: Path):
        """
        Generate the list of file names of the corpus, both in txt and pickle format.
        Define the directory location where to save the files and the txt-vocabulary Path/name.\n
        The file names are extracted from the index file, also called "WARC file list",
        which MUST be one of those available at https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html.
        The index file is a list of URLs suffixes for the "News Dataset".
        The initialization generates valid URLs for the archives selected from the index such that
        the completed URLs point to downloadable archives in WARC format.

        :param warc_index: Path to the index file.
        :param range_start: Positional number of the first archive in the index to use.
        :param range_stop: Positional number of the last archive in the index to use.
        :param text_vocab: Path to text vocabulary file, if it exists it will be overwritten.
        :param corpus_dir: Path to the corpus directory, if it doesn't exist it will be created.
        """
        self.__warc_index: list[str] = []  # list of urls
        self.__text_corpus_files: list[Path] = []  # txt files
        self.__pkl_corpus_files: list[Path] = []  # pkl files
        self.__text_vocab: Path = text_vocab  # easily readable vocabulary
        self.__corpus_dir: Path = Path()
        self.set_corpus_dir(corpus_dir)
        self.__read_warc_index(warc_index, range_start, range_stop)

    def __read_warc_index(self, path: Path, istart: int, istop: int):
        """
        Populate the list of text and pickle file names using the warc index.

        :param path: Index Path.
        :param istart: First line to read.
        :param istop: Last line to read.
        """
        if path.is_file():
            with path.open('r') as index_file:
                lines = index_file.readlines()
                for i, arc in enumerate(lines):
                    if istart <= i <= istop:
                        arc = re.sub("\n", "", arc)
                        arc = f"https://data.commoncrawl.org/{arc}"
                        self.__warc_index.append(arc)
            logger.info('Warc index generated.')
            self.__corpus_files()  # generate corpus text and pickle file names
        else:
            logger.error("File not found")
            raise FileNotFoundError

    def __corpus_files(self):
        """
        Generate valid Paths for the corpus text and pickle files, using the index as reference.
        """
        for url_index in self.warc_index:
            path = self.__corpus_dir / Path(f'{re.search(r"CC-NEWS-[0-9-]+", url_index).group(0)}.txt')
            path_pkl = self.__corpus_dir / Path(f'{re.search(r"CC-NEWS-[0-9-]+", url_index).group(0)}.pkl')
            self.__text_corpus_files.append(path)
            self.__pkl_corpus_files.append(path_pkl)
        logger.info('Corpus file names generated.')

    @property
    def warc_index(self) -> list[str]:
        """
        Get the list of the URLs of the index.\n
        The URLs point to downloadable warc archives from https://data.commoncrawl.org/.

        :return: List of URLs
        """
        return self.__warc_index

    @property
    def text_corpus_files(self) -> list[Path]:
        """
        Get the list of the Paths for the text files of the corpus.

        :return: List of Paths of text files.
        """
        return self.__text_corpus_files

    @property
    def pkl_corpus_files(self) -> list[Path]:
        """
        Get the list of the Paths for the pickle files of the corpus.

        :return: List of Paths of pickle files.
        """
        return self.__pkl_corpus_files

    @property
    def text_vocab(self) -> Path:
        """
        Get the Path of the vocabulary for the corpus.

        :return: Path to the vocabulary.
        """
        return self.__text_vocab

    @property
    def corpus_dir(self) -> Path:
        """
        Get the Path of the corpus directory.

        :return: Path of the corpus directory.
        """
        return self.__corpus_dir

    def set_corpus_dir(self, dir_path: Path):
        """
        Set the Path to the directory where the corpus will be saved.
        Create the directory if it does not exist.

        :param dir_path: Path to the corpus directory.
        """

        try:
            dir_path.mkdir()
        except FileExistsError:
            logger.info(f"Directory {dir_path} found.")
        self.__corpus_dir = dir_path
        logger.info(f"Text corpus directory set to {self.__corpus_dir}.")


class CorpusBuilder:
    """
    This class is responsible for generating a normalized text corpus from the News Dataset of the CommonCrawl project.
    It is possible to download, extract, normalize and serialize the corpus with using an instance of this class.
    """
    __stop_set = {'_', '%', 'p', 've', 'm', 'up', 'k', 'a', 'p', 't', 'v', 'r', 'g', 'h', 'q', 'z', 'b', 'f',
                  'd', 'o', 'num', 'news', 'say', 'get', 'year', 'share', 'make', 'comment', 'people', 'week', 'month',
                  'hour', 'name', 'th', 'of', '°', 'an', 'at', 'not', 'this', 'as', 'to', 'go', 're', 'come', 'thing',
                  'day', 'time', 'do', 'am', 'pm', 'co', 'have', 'take', 'way', 'end'}
    __pos_set = {
        # "ADJ",      #: "adjective"
        # "ADP",      #: "adposition",
        # "ADV",      #: "adverb",
        # "AUX",      #: "auxiliary",
        # "CONJ",     #: "conjunction",
        # "CCONJ",    #: "coordinating conjunction",
        # "DET",      #: "determiner",
        # "INTJ",     #: "interjection",
        "NOUN",  #: "noun",
        # "NUM",      #: "numeral",
        # "PART",     #: "particle",
        # "PRON",     #: "pronoun",
        # "PROPN",    #: "proper noun",
        # "PUNCT",    #: "punctuation",
        # "SCONJ",    #: "subordinating conjunction",
        # "SYM",      #: "symbol",
        # "VERB",  #: "verb",
        # "X",        #: "other",
        # "EOL",      #: "end of line",
        # "SPACE"     #: "space",
    }

    def __init__(self, data_paths: DataPaths):
        """
        Initialize a new :class:`CorpusBuilder` instance.\n
        To load a previous instance, call :meth:`~CorpusBuilder.load_status()` followed by
        :meth:`~CorpusBuilder.load_corpus()`.

        :param data_paths: :class:`DataPaths` instance, metadata for the environment of the corpus.
        """
        self.__paths: DataPaths = data_paths
        self.__status_path: Path = Path()  # Path
        self.__corpus: list = []
        self.__split: bool = False  # If True, the documents in the corpus are lists of words.
        self.__vocab: defaultdict[int]
        self.__word_count = 0

    def generate_texts_from_index(self, check_local_archive=True, keep_archive=True, check_txt=True, min_len=50):
        """
        Download the warc archives and extract their content into text files in the corpus directory.

        :param check_local_archive: If true, download only the archives that have not been already downloaded.
        :param keep_archive: Do not delete the archives after extraction.
        :param check_txt: If True, download and extract only the archives for which there is not a corresponding
                          text file in the corpus directory.
        :param min_len: Minimum document length in number of words.
        """

        if len(self.__paths.warc_index) == 0:
            logger.error("Warc index not defined!")
            raise ValueError
        for _index, archive_url in enumerate(self.__paths.warc_index):
            _a_name = re.search(r"CC-NEWS-[0-9-]+\.warc\.gz", archive_url).group(0)

            archive_name = self.data_paths.corpus_dir / Path(_a_name)
            txt_file_name = self.__paths.text_corpus_files[_index]

            download_archive = partial(wget.download, url=archive_url, out=str(self.data_paths.corpus_dir))
            extract_text = partial(CorpusBuilder.extract_text_from_archive,
                                   archive=archive_name,
                                   text_path=txt_file_name,
                                   min_len=min_len)
            deduplicate_lines = partial(CorpusBuilder.deduplicate_lines_in_txt, source=txt_file_name)

            if check_txt:  # look for the text file
                if txt_file_name.is_file():
                    logger.info(f"File {txt_file_name} found.")
                else:
                    logger.info(f"File {txt_file_name} not found.")
                    if not archive_name.is_file():
                        logger.info(f"Archive not found, downloading: {archive_url}")
                        download_archive()
                    extract_text()
                    deduplicate_lines()
            elif check_local_archive:  # look for the archive
                if not archive_name.is_file():
                    logger.info(f"Archive not found, downloading: {archive_url}")
                    download_archive()
                extract_text()
                deduplicate_lines()
            else:  # start from scratch
                logger.info(f"Downloading archive: {archive_url}")
                download_archive()
                extract_text()
                deduplicate_lines()

            # delete archive if present
            if not keep_archive and archive_name.is_file():
                archive_name.unlink()
                logger.info(f"Deleted archive: {archive_name}")

    @staticmethod
    def extract_text_from_archive(archive: Path, text_path: Path, min_len=50):
        """
        Extract the text documents from a warc archive and save it in a text file.

        Each line in the text file represents a document.

        :param archive: Archive to process.
        :param text_path: Text file to write to.
        :param min_len: Threshold on the minimum number of words, discard document otherwise.
        """
        with archive.open('rb') as stream, \
                text_path.open('w') as processed:
            logger.info(f"\n\nExtracting {archive}, writing to {text_path}\n")
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    html = record.content_stream().read()
                    resp = trafilatura.bare_extraction(filecontent=html,
                                                       output_format="txt",
                                                       target_language="en")
                    if resp is not None:
                        text = re.sub(r'[\n\r\v\f]', r' ', resp["raw_text"])
                        text = re.sub(r' +', r' ', text)
                        if len(text.split(" ")) > min_len:
                            processed.write(f'{text}\n')
            logger.info(f"Extraction of {archive} completed.")

    def load_and_clean_origin(self,
                              split=True,
                              pos: set[str] = None,
                              stop: set[str] = None,
                              only_missing: bool = False,
                              n_process: int = -1):
        """
        Process the raw text corpus and generate a normalized corpus.
        Save the normalized corpus in pickle files and generate a vocabulary for it.

        :param split: If True, the documents are saved as list of words.
        :param pos: Part of Speech set, if None use class defined one.
        :param stop: Stopword set, if None use class defined one.
        :param only_missing: If True, process only the text files for which there is not a corresponding pickle file.
        :param n_process: Number of cores to use.
        """
        if pos is None:
            pos = CorpusBuilder.__pos_set
        if stop is None:
            stop = CorpusBuilder.__stop_set

        self.__split = split

        existing_ones = []
        if only_missing:
            to_process = []
            to_process_ids = []
            for f_id, path in enumerate(self.__paths.text_corpus_files):
                if not self.__paths.pkl_corpus_files[f_id].is_file():
                    to_process.append(path)
                    to_process_ids.append(f_id)
            existing_ones = [self.__paths.pkl_corpus_files[f_id]
                             for f_id, _ in enumerate(self.__paths.text_corpus_files)
                             if self.__paths.pkl_corpus_files[f_id].is_file()]
            if len(existing_ones) > 0:
                logger.info("Reloading pickle files.")
                self.load_corpus(files=existing_ones)
        else:
            to_process = self.__paths.text_corpus_files
            to_process_ids = [*range(len(to_process))]

        logger.info("Loading spacy model.")
        nlp = spacy.load("en_core_web_md", disable=['ner'])
        for c_id, corpus in enumerate(CorpusBuilder.__corpus_iterator(to_process)):
            logger.info("Spacy pipe.")
            docs = nlp.pipe(texts=corpus, batch_size=128, n_process=n_process)
            _docs = []
            for doc in docs:
                language = detect(doc.text)
                if language['lang'] == 'en' and language['score'] > 0.90:
                    _docs.append(" ".join([t.lemma_ for t in doc if t.pos_ in pos]))
            docs = _docs
            logger.info("Pipe completed.")
            logger.info("Regex cleaning.")
            docs = [[word for word in CorpusBuilder.clean_doc(doc).split(" ") if word not in stop]
                    for doc in docs]
            docs = [doc for doc in docs if doc != ['']]
            pkl_file = self.__paths.pkl_corpus_files[to_process_ids[c_id]]  # get pickle file name
            self.__corpus = [*self.__corpus, *docs]
            with pkl_file.open('wb') as file:
                pickle.dump(docs, file)
            logger.info(f"{pkl_file} saved.")

        _corpus = Path("_corpus_temp_")
        with _corpus.open('w') as file:
            for doc in self.__corpus:
                file.write(f"{doc}\n")
        CorpusBuilder.deduplicate_lines_in_txt(_corpus)
        with _corpus.open('r') as file:
            lines = file.readlines()
            if not split:
                self.__corpus = [" ".join(re.sub("\n", "", line)) for line in lines]
        _corpus.unlink()  # remove temporary corpus file

        logger.info("Done cleaning.")
        self.__make_vocab()

    def __make_vocab(self):
        """
        Generate a txt vocabulary of words and their count sorted by most to least frequent.
        The first line in the vocabulary is the total word count.
        """
        logger.info("Making vocabulary with counts.")
        self.__vocab = defaultdict(int)
        for line in self.__corpus:
            if self.__split:
                words = line
            else:
                words = line.split(" ")
            for word in words:
                self.__vocab[word] += 1

        self.__vocab = {k: v for k, v in sorted(self.__vocab.items(), key=lambda entry: entry[1], reverse=True)}
        logger.info(f"Vocabulary generated, {len(self.__vocab.keys())} entries.")
        self.__word_count = sum(self.__vocab.values())
        logger.info(f"{self.__word_count} total tokens.")

        with open(self.__paths.text_vocab, 'w') as voc:
            voc.write(f'{self.__word_count}\n')
            for k, v in self.__vocab.items():
                voc.write(f'{k} {v}\n')
            logger.info(f"Text vocabulary saved to {self.__paths.text_vocab}.")

    def save_vocab(self):
        """
        Generate the vocabulary from the current corpus and save it in a text file.
        """
        if self.__corpus:
            self.__make_vocab()
        else:
            logger.warning("Corpus not found! Cannot compute and save the vocabulary.")

    def set_status_path(self, path: Path):
        """
        Set the Path name for saving/loading a serialized instance of this class, without the corpus.\n
        Setting the Path does not save the instance, use :meth:`~CorpusBuilder.save_status()` for that.

        :param path: Status Path.
        """
        if self.status_path == path:
            return self
        self.__status_path = path
        logger.info(f"Status path updated to {self.__status_path}.")

    def save_status(self):
        """
        Save the serialized instance of this class in a pickle file, without the corpus.
        """
        corpus = self.__corpus
        with self.__status_path.open('wb') as dm:
            self.__corpus = []  # save without corpus
            pickle.dump(self, dm, pickle.HIGHEST_PROTOCOL)
            self.__corpus = corpus
            logger.info(f"Status saved {self.__status_path}.")

    @classmethod
    def load_status(cls, path: Path) -> Self:
        """
        Retrieve a previously serialized instance of this class.\n
        The corpus must be loaded separately with :meth:`~CorpusBuilder.load_corpus()`.

        :param path: Path to status file.
        :return: Instance of this class, without the corpus.
        """
        with path.open('rb') as dm:
            logger.info(f"Loading status {path}.")
            return pickle.load(dm)

    def load_corpus(self, files: list[Path] = None):
        """
        Load the corpus from the list of pickle files passed as parameter.\n
        If no list is passed as parameter, load the corpus from the pickle files defined in
        :attr:`~CorpusBuilder.data_paths` during initialization of the instance.

        :param files: List of Paths of pickle corpus file to load.
        """
        if files is None:
            files = self.__paths.pkl_corpus_files
        for path in files:
            with path.open('rb') as file:
                self.__corpus = [*self.__corpus, *pickle.load(file)]

    @property
    def status_path(self) -> Path:
        """
        Get the Path to use for serialization of an instance of this class.

        :return: Path to the :class:`CorpusBuilder` instance.
        """
        return self.__status_path

    @property
    def data_paths(self) -> DataPaths:
        """
        Get the :class:`DataPaths` instance of the current instance.\n
        DataPaths is a collection of metadata for the current :class:`CorpusBuilder` instance.

        :return: :class:`DataPaths` instance.
        """
        return self.__paths

    @property
    def documents_are_split_in_words(self) -> bool:
        """
        True if :attr:`corpus` is a list of lists of words.

        :return: Boolean.
        """
        return self.__split

    @property
    def corpus(self):
        """
        Get the corpus of the current instance.
        The corpus can be a list of strings or a list of lists of words (default).

        :return: List of documents.
        """
        return self.__corpus

    @staticmethod
    def __corpus_iterator(paths: list[Path] | Path) -> Iterator[str]:
        """
        Iterate over a list of text files and yield their content one at a time.

        :param paths: List of text files Paths, or single Path.
        """
        if paths.__class__ is not list:
            paths = [paths]  # wrap inside list
        for path in paths:
            corpus = []
            with path.open('r') as file:
                lines = file.readlines()
                for line in lines:
                    line = re.sub(r'\n', r'', line)
                    corpus.append(line)
                yield corpus

    @staticmethod
    def clean_doc(doc: str) -> str:
        """
        Perform a sequence of regex substitutions on the given document.

        :param doc: Document to clean.
        :return: Document cleaned.
        """
        # unicode values available at https://www.unicodepedia.com/groups/
        doc = re.sub(u'[^\U00000021-\U0000024F]', r' ', doc.lower())
        doc = re.sub(r'(https?|www)[^ ]*', r' ', doc)  # remove links

        doc = re.sub(r'( ?("\'?)|\'?" | \'+|^\'|'
                     r' ?\(+| ?\)+)', r' ', doc)
        doc = re.sub(r'(^|\t| )#', r' ', doc)
        doc = re.sub(r' ?@[a-zA-Z0-9_-]+[.,]? ?', r' ', doc)  # remove usernames
        doc = re.sub(r'@[^ ]*', r' ', doc)

        doc = re.sub(r'([.|,]?[0-9]+[.|,]?)+', r' ', doc)  # remove numeric values

        doc = re.sub(r'[\[\]{}«»¼½¾©®×§¹²³º÷~;`^¡&%'
                     r'\U000000A0:/\-"#,<>´=\\|*!+?.·\']'
                     r'|_{2,}', r' ', doc)  # remove specific characters
        doc = re.sub(r'[a-z]+@[a-z]+?', r' ', doc)  # rests of emails
        doc = re.sub(r' +', r' ', doc)
        doc = re.sub(r'(^( )+)|(( )+$)', r'', doc)  # remove leading or trailing multiple spaces.

        return doc

    @staticmethod
    def doc_from_link(link: str) -> str | None:
        """
        Download and extract the text body of a web page from the given URL, return None if problems during download or
        extraction arise.

        :param link: URL link of the web page.
        :return: Raw text body as a string or None if the document cannot be extracted.
        """
        logger.info(f"Fetching html from {link}...")
        html = trafilatura.fetch_url(url=link)
        logger.info(f"Html fetched, starting extraction...")
        try:
            obj = trafilatura.bare_extraction(filecontent=html, output_format="txt", target_language="en")
        except ValueError:
            logger.error("Error extracting the text from the web page.")
            return None
        if obj is not None:
            return obj["raw_text"]
        return None

    @staticmethod
    def deduplicate_lines_in_txt(source: Path):
        """
        Deduplicate lines in the source text file.
        This method uses awk to perform deduplication.

        :param source: Path to text file.
        """
        target = "_temp_txt_"
        subprocess.run([f"awk '!visited[$0]++' {source} > {target}"], shell=True)
        subprocess.run([f"rm {source}"], shell=True)
        subprocess.run([f"mv {target} {source}"], shell=True)


if __name__ == '__main__':
    pass
