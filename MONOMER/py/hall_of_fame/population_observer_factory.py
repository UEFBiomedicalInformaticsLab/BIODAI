from abc import ABC, abstractmethod

from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hof_by_sum import HofBySum
from hall_of_fame.pareto_front import ParetoFront
from hall_of_fame.population_observer import PopulationObserver
from util.named import Named


class PopulationObserverFactory(ABC):

    @abstractmethod
    def create_population_observer(self) -> PopulationObserver:
        raise NotImplementedError()


class HallOfFameFactory(PopulationObserverFactory, Named):

    @abstractmethod
    def create_population_observer(self) -> HallOfFame:
        raise NotImplementedError()


class HofBySumFactory(HallOfFameFactory):
    __size: int

    def __init__(self, size: int):
        self.__size = size

    def create_population_observer(self) -> HofBySum:
        return HofBySum(size=self.__size)

    def name(self) -> str:
        return "Best " + str(self.__size) + " by sum factory"


class ParetoFrontFactory(HallOfFameFactory):

    def create_population_observer(self) -> ParetoFront:
        return ParetoFront()

    def name(self) -> str:
        return "Pareto front factory"
