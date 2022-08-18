from collections.abc import Iterable
from copy import copy


class Summer:

    def add(self, x: float) -> None:
        raise NotImplementedError()

    def add_all(self, elems: Iterable):
        for e in elems:
            self.add(e)

    def subtract(self, x: float) -> None:
        raise NotImplementedError()

    def get_sum(self) -> float:
        raise NotImplementedError()

    @classmethod
    def create(cls):
        """Creates a new summer."""
        raise NotImplementedError()

    @classmethod
    def sum(cls, elems):
        summer = cls.create()
        summer.add_all(elems)
        return summer.get_sum()

    @classmethod
    def mean(cls, elems):
        summer = cls.create()
        summer.add_all(elems)
        return summer.get_sum()/len(elems)


class NaiveSummer(Summer):

    __sum: float

    def __init__(self, tot=0.0):
        self.__sum = tot

    def add(self, x: float) -> None:
        self.__sum += x

    def subtract(self, x: float) -> None:
        self.__sum -= x

    def get_sum(self) -> float:
        return self.__sum

    def __copy__(self):
        return NaiveSummer(tot=self.__sum)

    def __deepcopy__(self):
        return copy(self)

    def __str__(self):
        return "Naive summer with sum: " + str(self.__sum)

    @classmethod
    def create(cls) -> Summer:
        return NaiveSummer()


class KahanSummer(Summer):
    __sum: float
    __c: float

    def __init__(self, tot=0.0, c=0.0):
        self.__sum = tot
        self.__c = c

    def add(self, x: float) -> None:
        y = x - self.__c
        t = self.__sum + y  # Alas, sum is big, y small, so low-order digits of y are lost.
        self.__c = (t - self.__sum) - y  # (t - sum) recovers the high-order part of y;
        # subtracting y recovers -(low part of y)
        self.__sum = t  # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
        # Next time around, the lost low part will be added to y in a fresh attempt.

    def subtract(self, x: float) -> None:
        y = x + self.__c
        t = self.__sum - y
        self.__c = (t - self.__sum) + y
        self.__sum = t

    def get_sum(self) -> float:
        return self.__sum

    def __copy__(self):
        return KahanSummer(tot=self.__sum, c=self.__c)

    def __deepcopy__(self):
        return copy(self)

    def __str__(self):
        return "Kahan summer with sum: " + str(self.__sum) + " c: " + str(self.__c)

    @classmethod
    def create(cls) -> Summer:
        return KahanSummer()
