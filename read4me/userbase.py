"""
**What is this module for?**

This module provides a single entry point to manage the user reading preferences
for the Read4Me telegram bot.

**What classes are there?**

* :class:`UserBase` - Singleton, for storing the user preferences, serializable.
* :class:`UserPreferences` - User reading preferences and other data.

**How can you use it?**

.. note::
	Each reference of `document` is intended as a **processed** instance of the class :class:`~read4me.docprocessing.Doc`.

Initialize a :class:`UserBase` instance using a shared list of topics among users.

.. code-block::

    userbase = UserBase(topics, topics_descriptions)

:attr:`~UserPreferences.user` - Add a new user to the userbase or get a specific one.

.. code-block::

    user: UserPreferences = userbase.user(user_id)

:meth:`~UserBase.predict_interest` - Score how much a user could be interested in reading a document.

.. code-block::

    userbase.predict_interest(doc, user_id)

:meth:`~UserBase.score_custom_topics` - Find which custom topics are most similar to a given document.

.. code-block::

    userbase.score_custom_topics(doc, user_id)

:meth:`~UserBase.update_user_preferences` - Update the reading preferences of a user using a document and a score for it.

.. code-block::

    userbase.update_user_preferences(user_id, doc, vote)

:meth:`~UserBase.set_topics` - Change the shared topics in the userbase, this will affect all UserPreferences
stored in the userbase so that their preferences reflect the new topics.

.. code-block::

    userbase.set_topics(new_topics, new_topics_descriptions)

.. warning::

    Change in the Word2Vec model is not yet supported and requires manually deleting the bot data and starting over.
"""

import numpy
from numpy.typing import NDArray

from read4me.docprocessing import Doc


class UserPreferences:
    """
    This class is a container for the data of a single user.
    All instances of this class are meant to be stored in a :class:`UserBase` instance.

    The user data is composed of a user id, a user vector and a list of custom topics with a list of their descriptions.

    The :attr:`~UserPreferences.user_vector` is a numpy array with dimensionality equal to the number of topics stored
    in :attr:`UserBase.topics`.
    Each weight stored in the :attr:`~UserPreferences.user_vector` is positionally related to the topics stored in
    :attr:`UserBase.topics`.

    A custom topic is defined by the user at runtime and is generated as a combination of semantic vectors
    and stored as a numpy array in :attr:`~UserPreferences.custom_topics`.
    For each custom topic is also stored its description in :attr:`~UserPreferences.custom_topics_descriptions`
    as a string.
    """

    def __init__(self, user_id: int, num_topics: int):
        """
        Initialize an instance for a new user.

        :param user_id: Id of the user.
        :param num_topics: Number of topics, defines the number of dimensions for the user vector.
        """
        self.__user_id: int = user_id
        self.__user_vector: NDArray = numpy.zeros(num_topics)
        self.__updates_counter = 0

        # custom user topics
        self.__custom_topics: list[NDArray] = []
        self.__custom_topics_descriptions: list[str] = []

    @property
    def user_id(self) -> int:
        """
        Get the user id of the current instance.

        :return: User id.
        """
        return self.__user_id

    @property
    def user_vector(self) -> NDArray:
        """
        :getter: Get the user vector.
        :setter: Set the user vector with a new NDArray vector during the learning phase.

        Use the setter only when the new vector and the current user vector refer to the same topics.
        Use :meth:`~UserPreferences.replace_user_vector` when the topics have changed and a new vector of preferences
        needs to be set.
        Every update to the user vector is intended as a step in learning the user preferences.
        This method uses a simple forgetting strategy of normalizing the user vector to a length of 5 every 10 updates.

        :return: User vector.
        """
        return self.__user_vector

    @user_vector.setter
    def user_vector(self, new_vec: NDArray):
        """
        Set the user vector with a new NDArray vector during the learning phase.

        Use the setter only when the new vector and the current user vector refer to the same topics.
        Use :meth:`~UserPreferences.replace_user_vector` when the topics have changed and a new vector of preferences
        needs to be set.

        Every update to the user vector is intended as a step in learning the user preferences.

        This method uses a simple forgetting strategy of normalizing the user vector to a length of 5 every 10 updates.

        :param new_vec: New user vector.
        """
        self.__user_vector = new_vec
        self.__updates_counter += 1
        if self.__updates_counter == 10:
            self.__user_vector = 5 * self.__user_vector / numpy.linalg.norm(self.__user_vector)
            self.__updates_counter = 0

    def has_custom_topics(self) -> bool:
        """
        Return True if the user has custom topics.

        :return: Boolean.
        """
        return bool(self.__custom_topics)

    def has_this_custom_topic(self, topic: NDArray) -> bool:
        """
        Check if the given topic is equal within a numerical tolerance
        to at least one of the custom topics of the user.

        :param topic: Semantic vector representing a topic.
        :return: Boolean.
        """
        return bool(numpy.asarray([numpy.isclose(topic, c_t).all() for c_t in self.__custom_topics]).any())

    def add_custom_topic(self, topic: NDArray, topic_description: str):
        """
        Add a custom topic and its description for the current user.

        :param topic: Semantic vector of a custom topic.
        :param topic_description: Custom topic description.
        """
        self.__custom_topics.append(topic)
        self.__custom_topics_descriptions.append(topic_description)

    def delete_custom_topic(self, topic_id: int):
        """
        Remove a custom topic and its description from the current user.

        The id refers to the position of the topic in the list of custom topics, starting from 0.

        :param topic_id: Position of the custom topic to remove.
        """
        del self.__custom_topics[topic_id]
        del self.__custom_topics_descriptions[topic_id]

    def delete_all_custom_topics(self):
        """
        Remove all user's custom topics.
        """
        self.__custom_topics = []
        self.__custom_topics_descriptions = []

    @property
    def custom_topics(self) -> list[NDArray]:
        """
        Get the full list of custom topics vectors of the current user.

        :return: List of NDArray vectors.
        """
        return self.__custom_topics

    @property
    def custom_topics_descriptions(self) -> list[str]:
        """
        Get the full list of custom topics descriptions of the current user.

        :return: List of topics descriptions.
        """
        return self.__custom_topics_descriptions

    @property
    def num_custom_topics(self) -> int:
        """
        Get the number of custom topics of the current user.

        :return: Number of custom topics of the user.
        """
        return len(self.__custom_topics)

    def replace_user_vector(self, new_vec: NDArray):
        """
        Replace the user vector with a new vector.

        Do not use this method for the learning phase, use the setter :meth:`~UserPreferences.user_vector` instead.
        The new vector can have a different number of dimensions.

        :param new_vec: New user preferences vector.
        """
        self.__user_vector = new_vec


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)  # instantiate unique instance
        return cls._instances[cls]  # retrieve single instance of that class


class UserBase(metaclass=MetaSingleton):
    """
    Singleton class for storing the :class:`UserPreferences` instances in a dictionary with their user ids as keys.
    The instance of this class stores the topic vectors and the topics descriptions that are shared by all users, also
    it performs a prediction of interest of a user in a document and updates the user preferences.
    """

    def __init__(self, topics: NDArray, topics_descriptions: list[tuple[int, list]]):
        """
        Initialize singleton instance with topics and their descriptions.

        :param topics: NDArray vectors of the topics.
        :param topics_descriptions: Description of the topics.
        """
        self.__current_topics: NDArray = topics
        self.__topics_descriptions: list[tuple[int, list[tuple[str, float]]]] = topics_descriptions
        self.__users: dict[int, UserPreferences] = dict()

    @property
    def users(self) -> dict[int, UserPreferences]:
        """
        Get all the users.

        :return: Dictionary of users.
        """
        return self.__users

    @property
    def topics(self) -> NDArray:
        """
        Get the shared topic vectors.

        :return: NDArray vectors.
        """
        return self.__current_topics

    @property
    def topics_descriptions(self) -> list[tuple[int, list[tuple[str, float]]]]:
        """
        Get the list of topics descriptions.

        Each topic description has a positional topic id and a list of tuples of the form <word, weight>.

        :return: List of topics descriptions.
        """
        return self.__topics_descriptions

    def set_topics(self, new_topics: NDArray, new_topics_descriptions: list[tuple[int, list[tuple[str, float]]]]):
        """
        Set the shared topics of the :class:`UserBase`.
        If the current topics and the new topics are different,
        remap all user's :attr:`UserPreferences.user_vector` to the new topics.

        :param new_topics: New topics for the UserBase.
        :param new_topics_descriptions: Descriptions of the new topics.
        """
        if self.topics.shape == new_topics.shape and (self.topics == new_topics).all():
            return

        conversion_matrix = new_topics @ numpy.transpose(self.__current_topics)
        for user_id in self.__users.keys():
            self.user(user_id).replace_user_vector(conversion_matrix @ self.user(user_id).user_vector)

        self.__current_topics = new_topics
        self.__topics_descriptions = new_topics_descriptions

    def user(self, user_id: int) -> UserPreferences:
        """
        Get, or create and get, the user with the specified id.

        :param user_id: The user id.
        """
        if user_id not in self.__users:
            self.__users[user_id] = UserPreferences(user_id, len(self.__current_topics))
        return self.__users[user_id]

    def update_user_preferences(self, user_id: int, doc: Doc, vote: float):
        """
        Update the preferences for a user, based on the vote left for the document.

        :param user_id: User id.
        :param doc: Processed :class:`~read4me.processing.Doc` instance.
        :param vote: User's vote for the document.
        """
        if vote == 0:
            return

        # halves the vote when similarity is low
        scaling = 1
        if doc.has_triggered_fallback:
            scaling /= 2
        scaled_doc_vec = scaling * vote * doc.vector()
        self.user(user_id).user_vector += scaled_doc_vec

    def predict_interest(self, doc: Doc, user_id: int) -> float:
        """
        Given a :class:`~read4me.Doc` document, compute a prediction score for the interest of the user in the document,
        the score is a number from -1 to 1.

        :param doc: Processed :class:`~read4me.Doc` document.
        :param user_id: Target user id.
        :return: Score from -1 to 1.
        """
        user_vector = self.user(user_id).user_vector
        if (user_vector == numpy.zeros(user_vector.shape[0])).all():
            return 0
        score = doc.vector().dot(user_vector)
        return score / (1 + abs(score))
        # return score/(math.sqrt(1+score**2))

    def score_custom_topics(self, doc: Doc, user_id: int, min_similarity: float = 0.55) -> list[tuple[int, str]]:
        """
        For each custom topic of a given user, get the descriptions of the custom topics
        that are most similar to the document.
        Return an empty list if there is no match.

        :param doc: :class:`~read4me.Doc` instance, processed using the custom topics of the user.
        :param user_id: Target user id.
        :param min_similarity: Lower similarity value.
        :return: List of tuples, each tuple is a custom topic id with its description.
        """
        counts = doc.vector()
        custom_desc = self.user(user_id).custom_topics_descriptions
        descriptions = [(cust_top_id, custom_desc[cust_top_id])
                        for cust_top_id in range(self.user(user_id).num_custom_topics)
                        if counts[cust_top_id] >= min_similarity]
        return descriptions


if __name__ == '__main__':
    pass
