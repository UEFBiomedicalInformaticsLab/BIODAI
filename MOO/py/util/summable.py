from abc import ABC, abstractmethod
from collections.abc import Sequence


class Summable(ABC):

    @abstractmethod
    def sum(self):
        raise NotImplementedError()


class SummableSequence(Summable, Sequence, ABC):
    pass
