from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

from individual.confident_individual import ConfidentIndividual
from individual.fit_individual_by_listlike import FitIndividualByListlike
from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.peculiar_individual import PeculiarIndividual
from model.model import Classifier
from util.hyperbox.hyperbox import Interval
from util.preconditions import check_none
from util.sequence_utils import sequence_to_string
from util.utils import IllegalStateError


class PeculiarIndividualByListlike(PeculiarIndividual, FitIndividualByListlike, ConfidentIndividual, ABC):
    __predictors: Sequence[Optional[Classifier]]
    __stats: dict
    __personalized_feature_importance: Optional[Sequence[float]]
    __std_dev: Sequence[Optional[float]]
    __ci95: Sequence[Optional[Interval]]
    __bootstrap_mean: Sequence[Optional[float]]

    def __init__(self, n_objectives: int, seq=()):
        FitIndividualByListlike.__init__(self, fitness=PeculiarFitness(n_objectives=n_objectives), seq=seq)
        self.__stats = {}  # dictionary of key -> stat_value
        self.__predictors = [None]*n_objectives
        self.__personalized_feature_importance = None
        self.__std_dev = [None]*n_objectives
        self.__ci95 = [None]*n_objectives
        self.__bootstrap_mean = [None]*n_objectives

    def set_stats(self, stats):
        """Previous prediction_stats are removed."""
        self.__stats = stats

    def reset_stats(self):
        """Previous prediction_stats are removed."""
        self.__stats = {}

    def get_stat(self, name):
        return self.__stats[name]

    def get_stats(self) -> dict:
        """Object returned is a copy."""
        return self.__stats.copy()

    def set_predictors(self, predictors: Sequence[Classifier]):
        if len(predictors) != self.n_objectives():
            raise ValueError()
        self.__predictors = check_none(predictors)

    def get_predictors(self) -> Sequence[Classifier]:
        return self.__predictors

    def reset_predictors(self):
        self.__predictors = [None]*self.n_objectives()

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

    def set_personalized_feature_importance(self, personalized_feature_importance: Sequence[float]):
        if self.sum() != len(personalized_feature_importance):
            raise ValueError(
                "Passed feature importances are in a wrong number.\n" +
                "Passed importances: " + str(personalized_feature_importance) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__personalized_feature_importance = personalized_feature_importance

    def has_personalized_feature_importance(self) -> bool:
        return self.__personalized_feature_importance is not None

    def get_personalized_feature_importance(self) -> Sequence[float]:
        if self.has_personalized_feature_importance():
            return self.__personalized_feature_importance
        else:
            raise IllegalStateError()

    def reset_personalized_feature_importance(self):
        self.__personalized_feature_importance = None

    def set_std_dev(self, std_dev: Sequence[Optional[float]]):
        if len(std_dev) != self.n_objectives():
            raise ValueError(
                "Passed standard deviations are in a wrong number.\n" +
                "Passed standard deviations: " + str(std_dev) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__std_dev = std_dev

    def set_ci95(self, ci95: Sequence[Optional[Interval]]):
        if len(ci95) != self.n_objectives():
            raise ValueError(
                "Passed 95% confidence intervals are in a wrong number.\n" +
                "Passed intervals: " + str(ci95) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__ci95 = ci95

    def std_dev(self) -> Sequence[Optional[float]]:
        return self.__std_dev

    def ci95(self) -> Sequence[Optional[Interval]]:
        return self.__ci95

    def set_bootstrap_mean(self, bootstrap_mean: Sequence[Optional[float]]):
        if len(bootstrap_mean) != self.n_objectives():
            raise ValueError(
                "Passed bootstrap means are in a wrong number.\n" +
                "Passed bootstrap means: " + str(bootstrap_mean) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__bootstrap_mean = bootstrap_mean

    def bootstrap_mean(self) -> Sequence[Optional[float]]:
        return self.__bootstrap_mean

    def __str__(self) -> str:
        res = FitIndividualByListlike.__str__(self)
        res += "Standard deviations: " + sequence_to_string(li=self.__std_dev) + "\n"
        res += "Confidence intervals: " + sequence_to_string(li=self.__ci95) + "\n"
        res += "Bootstrap means: " + sequence_to_string(li=self.__bootstrap_mean) + "\n"
        if self.has_personalized_feature_importance():
            res += "Personalized feature importance: " + str(self.get_personalized_feature_importance()) + "\n"
        else:
            "No personalized feature importance."
        return res

    def mothball(self) -> PeculiarIndividualByListlike:
        res = deepcopy(self)
        res.reset_stats()
        res.reset_personalized_feature_importance()
        res.reset_predictors()
        return res
