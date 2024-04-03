from collections import Iterable
from typing import Optional

from util.summable import SummableSequence


class CachedTuple(tuple, SummableSequence):
    __sum: [Optional]

    def __new__(cls, x: Iterable):
        return super(CachedTuple, cls).__new__(cls, tuple(x))

    def __init__(self, x: Iterable):
        self.__sum = None

    def sum(self):
        if self.__sum is None:
            self.__sum = sum(self)
        return self.__sum
