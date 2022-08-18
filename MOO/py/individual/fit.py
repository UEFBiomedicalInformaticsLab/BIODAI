from abc import ABC, abstractmethod

from individual.fitness.high_best_fitness import HighBestFitness


class Fit(ABC):

    @abstractmethod
    def get_fitness(self) -> HighBestFitness:
        """Higher is better. The fitness returned is a copy."""
        raise NotImplementedError()
