from abc import abstractmethod, ABC
from collections.abc import Iterable


class PopulationObserver(ABC):

    @abstractmethod
    def update(self, new_elems: Iterable):
        raise NotImplementedError()

    def signal_final(self, final_elems: Iterable):
        pass
