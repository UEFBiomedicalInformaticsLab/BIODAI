from abc import ABC, abstractmethod

from individual.fitness.high_best_fitness import HighBestFitness
from util.hyperbox.hyperbox import Hyperbox0B, ConcreteHyperbox0B


class Fit(ABC):

    @abstractmethod
    def get_test_fitness(self) -> HighBestFitness:
        """Higher is better. The fitness returned is a copy."""
        raise NotImplementedError()

    def fitness_hyperbox(self) -> Hyperbox0B:
        return ConcreteHyperbox0B.create_by_b_vals(b_vals=self.get_test_fitness().values)
