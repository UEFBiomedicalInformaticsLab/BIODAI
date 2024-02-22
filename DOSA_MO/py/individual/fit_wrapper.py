from copy import deepcopy
from typing import Any

from individual.fit import Fit
from individual.fitness.high_best_fitness import HighBestFitness


class FitWrapper(Fit):
    """Equals and hash depend only on the inner object (we assume that the inner object determines the fitness).
    This wrapper can be used to temporarily attach a different fitness to a solution, for example to compute
    a hall of fame using performance on a different sample set."""
    __inner: Any
    __fitness: HighBestFitness

    def __init__(self, inner: Any, fitness: HighBestFitness):
        self.__inner = inner
        self.__fitness = fitness

    def get_test_fitness(self) -> HighBestFitness:
        return deepcopy(self.__fitness)

    def unwrap(self):
        return self.__inner

    def __eq__(self, other):
        if isinstance(other, FitWrapper):
            return self.__inner == other.__inner
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.__inner)