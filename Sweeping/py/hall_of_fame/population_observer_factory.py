from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy

from hall_of_fame.fronts import Fronts
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hof_by_sum import HofBySum
from hall_of_fame.last_pop_hof import LastPopHof
from hall_of_fame.pareto_front import ParetoFront
from hall_of_fame.participants import Participants
from hall_of_fame.population_observer import PopulationObserver
from hall_of_fame.union_hof import UnionHof
from util.named import Named


class PopulationObserverFactory(ABC):

    @abstractmethod
    def create_population_observer(self) -> PopulationObserver:
        raise NotImplementedError()


class HallOfFameFactory(PopulationObserverFactory, Named):

    @abstractmethod
    def create_population_observer(self) -> HallOfFame:
        raise NotImplementedError()

    def name(self) -> str:
        return self.create_population_observer().name() + " factory"


class HofBySumFactory(HallOfFameFactory):
    __size: int

    def __init__(self, size: int):
        self.__size = size

    def create_population_observer(self) -> HofBySum:
        return HofBySum(size=self.__size)


class ParetoFrontFactory(HallOfFameFactory):

    def create_population_observer(self) -> ParetoFront:
        return ParetoFront()


class LastPopFactory(HallOfFameFactory):

    def create_population_observer(self) -> LastPopHof:
        return LastPopHof()


class ParticipantsFactory(HallOfFameFactory):

    def create_population_observer(self) -> Participants:
        return Participants()


class HofUnionFactory(HallOfFameFactory):
    __inner: Sequence[HallOfFameFactory]

    def __init__(self, inner_factories: Sequence[HallOfFameFactory]):
        self.__inner = copy(inner_factories)

    def create_population_observer(self) -> UnionHof:
        return UnionHof(inner_hofs=[h.create_population_observer() for h in self.__inner])


class FrontsFactory(HallOfFameFactory):
    __number_of_fronts: int

    def __init__(self, number_of_fronts: int):
        self.__number_of_fronts = number_of_fronts

    def create_population_observer(self) -> Fronts:
        return Fronts(number_of_fronts=self.__number_of_fronts)
