from abc import ABC, abstractmethod
from copy import deepcopy

from sortedcontainers import SortedSet

from comparators import Comparator
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from util.sequence_utils import str_in_lines


class Pruner(ABC):

    @abstractmethod
    def prune_last(self, elems: SortedSet) -> bool:
        raise NotImplementedError()


class DummyPruner(Pruner):

    def prune_last(self, elems: SortedSet) -> bool:
        return False


class PruneAtSize(Pruner):
    __size: int

    def __init__(self, size: int):
        self.__size = size

    def prune_last(self, elems: SortedSet) -> bool:
        return len(elems) > self.__size


class HallOfFameByComparator(HallOfFame):
    __elems: SortedSet
    __pruner: Pruner

    def __init__(self, comparator: Comparator = None, pruner: Pruner = DummyPruner()):
        self.__pruner = pruner
        if comparator is None:
            self.__elems = SortedSet()
        else:
            self.__elems = SortedSet(key=comparator.to_key())

    def add(self, elem):
        """The new element is deepcopied."""
        self.__elems.add(deepcopy(elem))
        self.prune()

    def prune(self):
        while self.prune_last():
            self.__elems.pop()

    def prune_last(self) -> bool:
        return self.__pruner.prune_last(elems=self.__elems)

    def __str__(self) -> str:
        return str_in_lines(self.__elems)

    def hofers(self) -> Hofers:
        res = []
        for h in self.__elems:
            res.append(deepcopy(h))
        return Hofers(elems=res, name=self.name(), nick=self.nick())
