from abc import ABC

from individual.fit_individual_by_listlike import FitIndividualByListlike
from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.peculiar_individual import PeculiarIndividual
from model.model import Classifier
from util.preconditions import check_none


class PeculiarIndividualByListlike(PeculiarIndividual, FitIndividualByListlike, ABC):
    __predictors: [Classifier]
    __stats: dict

    def __init__(self, n_objectives: int, seq=()):
        FitIndividualByListlike.__init__(self, fitness=PeculiarFitness(n_objectives=n_objectives), seq=seq)
        self.__stats = {}  # dictionary of key -> stat_value
        self.__predictors = [None]*n_objectives

    def set_stats(self, stats):
        """Previous prediction_stats are removed."""
        self.__stats = stats

    def get_stat(self, name):
        return self.__stats[name]

    def get_stats(self) -> dict:
        """Object returned is a copy."""
        return self.__stats.copy()

    def set_predictors(self, predictors: [Classifier]):
        if len(predictors) != self.n_objectives():
            raise ValueError()
        self.__predictors = check_none(predictors)

    def get_predictors(self) -> [Classifier]:
        return self.__predictors

    def set_crowding_distance(self, crowding_distance):
        self.fitness.set_crowding_distance(crowding_distance)

    def get_crowding_distance(self):
        return self.fitness.get_crowding_distance()

    def set_peculiarity(self, peculiarity):
        self.fitness.set_peculiarity(peculiarity)

    def get_peculiarity(self):
        return self.fitness.get_peculiarity()

    def set_social_space(self, social_space):
        self.fitness.set_social_space(social_space)

    def get_social_space(self):
        return self.fitness.get_social_space()
