from __future__ import annotations
from abc import abstractmethod
from copy import deepcopy

from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.predictive_individual import PredictiveIndividual


class PeculiarIndividual(PredictiveIndividual):
    fitness: PeculiarFitness

    def __init__(self, n_objectives: int):
        PredictiveIndividual.__init__(self, fitness=PeculiarFitness(n_objectives=n_objectives))

    @abstractmethod
    def get_stat(self, name):
        raise NotImplementedError()

    @abstractmethod
    def get_stats(self):
        """Object returned is a copy."""
        raise NotImplementedError()

    @abstractmethod
    def get_crowding_distance(self):
        return self.fitness.get_crowding_distance()

    @abstractmethod
    def get_peculiarity(self):
        return self.fitness.get_peculiarity()

    @abstractmethod
    def get_social_space(self):
        return self.fitness.get_social_space()

    def mothball(self) -> PeculiarIndividual:
        return deepcopy(self)
