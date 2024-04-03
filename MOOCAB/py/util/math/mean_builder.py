from abc import ABC, abstractmethod
from collections.abc import Iterable

from util.math.summer import KahanSummer, Summer
from util.utils import IllegalStateError


class MeanBuilder(ABC):

    @abstractmethod
    def add(self, x: float):
        raise NotImplementedError()

    @abstractmethod
    def mean(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError()

    def has_mean(self) -> bool:
        """Has received at least 1 sample."""
        return self.num_samples() > 0

    def add_all(self, elems: Iterable[float]):
        for x in elems:
            self.add(x=x)


class KahanMeanBuilder(MeanBuilder):
    __summer: Summer
    __counter: int

    def __init__(self):
        self.__counter = 0
        self.__summer = KahanSummer()

    def add(self, x: float):
        self.__summer.add(x=x)
        self.__counter += 1

    def mean(self) -> float:
        if self.has_mean():
            return self.__summer.get_sum() / self.__counter
        else:
            raise IllegalStateError()

    def num_samples(self) -> int:
        return self.__counter
