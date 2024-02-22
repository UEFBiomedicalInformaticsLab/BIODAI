from copy import copy
from typing import Callable, Sequence

from comparators import Comparator
from cross_validation.multi_objective.optimizer.nsga.nsga_types import NSGA2_TYPE
from ga_runner.secondary_sorting_strategy.individual_attribute_manager.individual_attribute_manager import \
    IndividualAttributeManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual
from ga_components.sorter.nsga2_with_secondary_comparator import Nsga2WithSecondaryComparator
from ga_components.sorter.pop_sorter import PopSorter


class NSGA2WithSecondarySortingStrategy(PopSorter):
    __attribute_managers: list[IndividualAttributeManager]  # In order of execution
    __inner_sorter: PopSorter
    __name: str
    __nick: str

    def __init__(
            self,
            attribute_managers: list[IndividualAttributeManager],
            secondary_comparator: Comparator,
            name: str, nick: str):
        """Attribute managers are executed in the order they are passed in."""
        self.__attribute_managers = attribute_managers
        self.__inner_sorter = Nsga2WithSecondaryComparator(secondary_comparator=secondary_comparator)
        self.__name = name
        self.__nick = nick

    def sort(self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        for am in self.__attribute_managers:
            # Calls all attribute managers in order. They compute attributes like crowding distance or social space.
            am.compute(individuals=pop, hp_manager=hp_manager)
        return self.__inner_sorter.sort(pop=pop, hp_manager=hp_manager)

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick

    def __str__(self) -> str:
        return self.name()

    def basic_algorithm_nick(self) -> str:
        return NSGA2_TYPE.nick()

    def to_be_added_to_stats(self) -> dict[str, Callable[[PeculiarIndividual], float]]:
        res = copy(self.__inner_sorter.to_be_added_to_stats())
        for am in self.__attribute_managers:
            if am.add_to_stats():
                res[am.attribute_name()] = am.getter()
        return res
