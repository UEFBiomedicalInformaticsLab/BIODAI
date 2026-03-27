from __future__ import annotations
from abc import abstractmethod, ABC
from copy import deepcopy

from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.predictive_individual import PredictiveIndividualSV, PredictiveIndividual, PredictiveIndividualMV


class PeculiarIndividual(PredictiveIndividual, ABC):
    """A PeculiarIndividual has a PeculiarFitness. A peculiar fitness has social space,
    peculiarity and crowding distance (all Optional)."""
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

    def set_peculiarity(self, peculiarity):
        self.fitness.set_peculiarity(peculiarity)

    def get_peculiarity(self):
        return self.fitness.get_peculiarity()

    def set_crowding_distance(self, crowding_distance):
        self.fitness.set_crowding_distance(crowding_distance)

    def get_crowding_distance(self):
        return self.fitness.get_crowding_distance()

    def set_social_space(self, social_space):
        self.fitness.set_social_space(social_space)

    def get_social_space(self):
        return self.fitness.get_social_space()

    def mothball(self) -> PeculiarIndividual:
        return deepcopy(self)

    def get_test_fitness(self) -> PeculiarFitness:
        return deepcopy(self.fitness)


class PeculiarIndividualSV(PeculiarIndividual, PredictiveIndividualSV, ABC):

    def __init__(self, n_objectives: int):
        PredictiveIndividualSV.__init__(self, fitness=PeculiarFitness(n_objectives=n_objectives))

    def mothball(self) -> PeculiarIndividualSV:
        return deepcopy(self)


class PeculiarIndividualMV(PeculiarIndividual, PredictiveIndividualMV, ABC):

    def __init__(self, n_objectives: int):
        PredictiveIndividualMV.__init__(self, fitness=PeculiarFitness(n_objectives=n_objectives))

    def mothball(self) -> PeculiarIndividualMV:
        return deepcopy(self)
