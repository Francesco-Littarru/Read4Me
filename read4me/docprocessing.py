"""
**What is this module for?**

This module is for the semantic processing of documents using a list of topic vectors as reference.
The documents are analyzed at a sentence level.

**What classes are there?**

* :class:`Doc` - Document

**How can you use it?**

To use the class simply initialize a Doc instance with the document string

.. code-block::

    doc = Doc("The quick brown fox jumps over the lazy dog.")

then process the document using the following:

* A function that takes a string, clean the text and returns a string.
* A set of Part-of-Speech tags that defines which types of words to keep.
  (see here for tags: https://universaldependencies.org/u/pos/)
* Models:

	* :class:`~gensim.models.keyedvectors.KeyedVectors` compatible Word2Vec model with normalized vectors.
	* :class:`~gensim.models.tfidfmodel.TfidfModel` and :class:`~gensim.corpora.dictionary.Dictionary` for it.
	* `Spacy model <https://spacy.io/usage/models/>`_.

* the topics as numpy arrays, generated with the Word2Vec model.

.. code-block::

    doc.process(w2v, tfidf, dct, nlp, pos_set, topics)

After the processing, some of the info you can get from the doc instance are:

* The number of sentences :attr:`~Doc.num_sentences`.
* Most frequent topics :attr:`~Doc.topic_counts`.
* Most similar sentence to a specific topic :meth:`~Doc.excerpt`.
* The :attr:`~Doc.similarity_matrix` mapping each sentence to each topic with the similarity score.
* A :attr:`~Doc.vector` (numpy array) that represents the embedded document as coefficients of the topics.
"""

from collections import defaultdict
import re
from typing import Self, Callable

from ftlangdetect import detect
from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
import numpy
from numpy.typing import NDArray
import spacy

__MAX_SENTENCE_LEN__ = 4096


class Doc:
    """
    An instance of this class contains information about the semantic relationship between a document and
    a list of topics.

    The following are required to process an instance:

    * Word2vec model compatible with gensim's :class:`~gensim.models.keyedvectors.KeyedVectors`.
    * List of topics that belong to the same vector space of the word2vec model, as a list of numpy arrays.
    * Tfidf model generated with gensim's :class:`~gensim.models.tfidfmodel.TfidfModel`.
    * Dictionary for the tfidf model, generated with gensim's :class:`~gensim.corpora.dictionary.Dictionary`.
    * Spacy model for the english language.

    """

    def __init__(self, doc: str):
        """
        Initialize a new instance with a document.

        :param doc: Document as a string.
        """
        self.__doc: str = doc
        self.__sentence_coordinates: list[tuple[int, int]] = []  # list of offset and length values
        self.__processed = False
        self.__similarity_matrix: NDArray | None = None  # dimensions: sentences x topics, value: similarity
        self.__doc_topics: dict[int, list[tuple[int, float]]] = defaultdict(list)  # topic_id: list most similar sents
        self.__topic_counts: dict[int, int] = dict()
        self.__similarity_fallback: bool = False

    def process(self,
                w2v: KeyedVectors, tfidf: TfidfModel, dct: Dictionary, nlp: spacy.Language,
                pos_set: set[str],
                topics: NDArray,
                text_cleaner: Callable[[str], str] = lambda txt: txt,
                min_similarity: float = 0.6,
                topics_per_sentence: int = 2) -> Self | None:
        """
        Process the document.

        The steps are the following:

        #. The text is segmented in sentences, cleaned and normalized.
        #. The sentences are embedded in centroid vectors, using word2vec and tfidf models.
        #. The two matrices of centroids and topic vectors are multiplied, obtaining the matrix of similarities.
           The shape of the matrix is given by the number of sentences and the number of topics.
        #. Selection of the most frequent topics document-wise, by first mapping a fixed number of most similar topics
           for each sentence and then counting the most frequent topics from this mapping.

        :param w2v: Word2vec :class:`~gensim.models.keyedvectors.KeyedVectors` model.
        :param tfidf: :class:`~gensim.models.tfidfmodel.TfidfModel` model.
        :param dct: :class:`~gensim.corpora.dictionary.Dictionary` for the tfidf model.
        :param nlp: Spacy language model, loaded with `spacy.load() <https://spacy.io/usage/models/>`_.
        :param pos_set: A set of Part-of-Speech strings. See https://universaldependencies.org/u/pos/.
        :param topics: Topic vectors.
        :param text_cleaner: Cleaning function, any function that takes a string as an argument and returns a string.
        :param min_similarity: Similarity threshold for the mapping of topics with sentences.
        :param topics_per_sentence: Maximum number of topics that should be mapped to each sentence above the similarity
               threshold.
        :return: None if the document is not in english, otherwise return the instance.
        """

        doc = nlp(self.__doc)
        if detect(re.sub(r"\n", " ", doc.text))['lang'] != 'en':
            return None

        # 1) cleaning and normalization
        idx_doc = []
        for sent in doc.sents:
            self.__sentence_coordinates.append((sent.start_char, sent.end_char))
            sent_ = " ".join([w.lemma_ for w in sent if w.pos_ in pos_set])
            sent_ = text_cleaner(sent_)
            idx_sent = dct.doc2bow(sent_.split(" "))  # OOV words are removed here.
            if len(idx_sent) > 0:
                idx_doc.append(idx_sent)
        doc_tfidf = tfidf[idx_doc]

        # 2) semantic embedding
        centroids: list[NDArray] = []
        for sent_tfidf in doc_tfidf:
            centroid = numpy.sum(
                [w2v[dct[word_id]] * value for (word_id, value) in sent_tfidf
                 if word_id in dct and dct[word_id] in w2v.key_to_index],
                axis=0)
            if centroid.__class__ is numpy.float64:
                centroid = numpy.zeros(w2v.vector_size)
            else:
                centroid = centroid / numpy.linalg.norm(centroid)
            centroids.append(centroid)

        # 3) matrix multiplication
        self.__similarity_matrix: NDArray = centroids @ numpy.transpose(topics)

        # 4) similarity mapping and selection
        if topics.shape[0] < topics_per_sentence:
            topics_per_sentence = topics.shape[0]
        self.__map_topics_to_sentences(topics_per_sentence, min_similarity)
        self.__set_topic_counts()

        self.__processed = True
        return self

    def num_sentences(self) -> int:
        """
        Get the number of sentences in the document.

        :return: Number of sentences.
        """
        return len(self.__sentence_coordinates)

    @property
    def processed(self) -> bool:
        """
        Get True if the document has been processed, False otherwise.

        :return: Boolean flag.
        """
        return self.__processed

    @property
    def topic_counts(self) -> dict[int, int]:
        """
        Get the dict of topic counts in the document.

        Each key is a topic id, each value is the number of sentences found to be correlated to that topic.
        Each sentence has a maximum number of topics correlated to it, defined as parameter in :meth:`~Doc.process`.

        :return: dict of counts.
        """
        return self.__topic_counts

    @property
    def has_triggered_fallback(self) -> bool:
        """
        Get True if the similarity between topics and sentences has never reached the minimum value set for processing.

        :return: Boolean flag.
        """
        return self.__similarity_fallback

    @property
    def similarity_matrix(self) -> NDArray | None:
        """
        Get the similarity matrix.

        :return: NDArray after processing the document, None otherwise.
        """
        return self.__similarity_matrix

    def sentence(self, pos_sent: int) -> str | None:
        """
        Get the sentence at position pos_sent from the original text.
        If the sentence is longer than the maximum allowed, return the sentence truncated.

        :param pos_sent: Sentence position in the text, starting from 0.
        :return: Sentence as a string, None if the document has not been processed.
        """
        if self.processed:
            (start, end) = self.__sentence_coordinates[pos_sent]
            if end - start > __MAX_SENTENCE_LEN__:  # length failsafe
                end = start + __MAX_SENTENCE_LEN__
            return self.__doc[start: end]
        return None

    def __set_topic_counts(self, topn: int = 5):
        """
        Populate a dict with the topn most frequent topics in the text, along with their counts.

        :param topn: Maximum number of topics to use.
        """
        highest_counts = sorted([(t_id, len(self.__doc_topics[t_id])) for t_id in self.__doc_topics.keys()],
                                key=lambda x: x[1], reverse=True)[:topn]
        self.__topic_counts = dict(highest_counts)

    def __map_topics_to_sentences(self, topics_per_sentence: int = 2, min_similarity: float = 0.60):
        """
        Select the most frequent topics document-wise, by first mapping a fixed number of most similar topics
        for each sentence and then counting the most frequent topics from this mapping.

        :param topics_per_sentence: Value to limit the number of topics linked to each sentence.
        :param min_similarity: Minimum similarity value.
        """

        for i, sentence_similarities in enumerate(self.__similarity_matrix):  # dimensions: sentences x topics
            # select the indexes (default 2) of the topics with the highest similarity to the i-th sentence
            tops = [*reversed(numpy.argsort(sentence_similarities)[-topics_per_sentence:])]
            for j in range(topics_per_sentence):
                # if the values exceed min_similarity, append the sentence_id and the value to the entry of the topic.
                if self.__similarity_matrix[i][tops[j]] >= min_similarity:
                    self.__doc_topics[tops[j]].append((i, self.__similarity_matrix[i][tops[j]]))

        # strategy for when the minimum similarity value is never met, trigger similarity fallback.
        if not self.__doc_topics:
            self.__similarity_fallback = True
            for i, sentence_similarities in enumerate(self.__similarity_matrix):
                max_sim_topic = int(numpy.argmax(sentence_similarities))
                self.__doc_topics[max_sim_topic].append((i, self.__similarity_matrix[i][max_sim_topic]))

    def excerpt(self, topic_id: int) -> str:
        """
        Given a topic id, get the most similar sentence to it.

        :param topic_id: Id of the topic, starting from 0.
        :return: Sentence as a string.
        """
        return self.sentence(self.most_similar_sentence_id(topic_id))

    def most_similar_sentence_id(self, topic_id: int) -> int:
        """
        Given a topic id, get the id of the most similar sentence to it.

        :param topic_id: Id of the topic, starting from 0.
        :return: Sentence id.
        """
        return int(numpy.argmax(self.__similarity_matrix[:, topic_id]))

    def vector(self) -> NDArray | None:
        """
        Create a normalized vector of the document, mapping dimensions to topics.
        The number of dimensions in the vector is equal to the number of topics.

        :return: Numpy NDArray vector.
        """
        counts = self.topic_counts
        if not self.processed or len(counts) == 0:
            return None
        vec = numpy.zeros(self.__similarity_matrix.shape[1])
        for k in counts.keys():
            vec[k] = counts[k]
        vec = vec / numpy.linalg.norm(vec)
        return vec


if __name__ == '__main__':
    pass
