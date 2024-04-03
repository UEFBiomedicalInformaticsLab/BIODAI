from abc import ABC
from functools import cmp_to_key

from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist

from comparators import Comparator, ComparatorOnCrowdingDistance
from cross_validation.multi_objective.optimizer.nsga.nsga_types import NSGA2_TYPE
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual
from ga_components.sorter.pop_sorter import PopSorter


class Nsga2WithSecondaryComparator(PopSorter, ABC):
    __secondary_comparator: Comparator

    def __init__(self, secondary_comparator: Comparator):
        self.__secondary_comparator = secondary_comparator

    def sort(self, pop: [PeculiarIndividual], hp_manager: HyperparamManager) -> [PeculiarIndividual]:
        pareto_fronts = sortNondominated(individuals=pop, k=len(pop))
        res = []
        for front in pareto_fronts:
            assignCrowdingDist(front)
            sorted_front = sorted(front, key=cmp_to_key(self.__secondary_comparator.compare))
            res.extend(sorted_front)
        return res

    def nick(self) -> str:
        return "NSGA2+" + self.__secondary_comparator.name()

    def name(self) -> str:
        return "NSGA2 with " + self.__secondary_comparator.name()

    def __str__(self) -> str:
        return self.name()

    def basic_algorithm_nick(self) -> str:
        return NSGA2_TYPE.nick()


class Nsga2WithCrowdingDistance(Nsga2WithSecondaryComparator):

    def __init__(self):
        Nsga2WithSecondaryComparator.__init__(self=self, secondary_comparator=ComparatorOnCrowdingDistance())

    def nick(self) -> str:
        return "Crowd"

    def name(self) -> str:
        return "NSGA2 with crowding distance"

    def __str__(self) -> str:
        return self.name()
