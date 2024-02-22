from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Iterable

from sortedcontainers import SortedSet

from util.summable import SummableSequence


class ListLikeIterator:

    def __init__(self, lst):
        self.__i = 0
        self.__lst = lst

    def __next__(self):
        if self.__i < self.__lst.__len__():  # Calling the method directly seems to be faster.
            res = self.__lst.__getitem__(self.__i)
            self.__i += 1
            return res
        else:
            raise StopIteration  # Does not seem to need parentheses.


class ListLike(SummableSequence, ABC):

    @abstractmethod
    def __init__(self, seq=()):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError()

    @abstractmethod
    def extend(self, iterable: Iterable):
        raise NotImplementedError()

    def __iter__(self):
        return ListLikeIterator(lst=self)

    @abstractmethod
    def to_numpy(self):
        raise NotImplementedError()

    @abstractmethod
    def sum(self):
        raise NotImplementedError()

    @abstractmethod
    def append(self, value):
        """Appends just one element."""
        raise NotImplementedError()

    def true_positions(self) -> SortedSet[int]:
        """Specific subclasses (e.g. sparse) can override with faster algorithms.
        TODO We can provide also another method that returns an iterable object that iterates with constant memory."""
        res = SortedSet()
        for i in range(len(self)):
            if self[i]:
                res.add(i)
        return res

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        self_len = len(self)
        if isinstance(other, Sized):
            if len(other) != self_len:
                return False
        self_iter = iter(self)
        other_iter = iter(other)
        for i in range(self_len):
            self_next = next(self_iter)
            try:
                other_next = next(other_iter)
            except StopIteration:
                return False
            if not self_next == other_next:  # Since it is not guaranteed that != is the same as not ==
                return False
        try:
            next(other_iter)
            return False  # other has too many elements.
        except StopIteration:
            return True

    def __ne__(self, other) -> bool:
        return not self == other
