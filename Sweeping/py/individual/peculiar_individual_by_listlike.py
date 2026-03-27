from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

from individual.confident_individual import ConfidentIndividual
from individual.fit_individual_by_listlike import FitIndividualByListlike
from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.peculiar_individual import PeculiarIndividual
from model.sv_model import Predictor
from util.preconditions import check_none
from util.str_utils import iterable_to_string
from util.utils import IllegalStateError


class PeculiarIndividualByListlike(
    ConfidentIndividual,
    PeculiarIndividual,             # (PredictiveIndividual)
    FitIndividualByListlike,
    ABC):
    __predictors: Sequence[Optional[Predictor]]
    __stats: dict
    __personalized_feature_importance: Optional[Sequence[float]]

    def __init__(self, n_objectives: int, seq=()):
        if hasattr(self, "fitness"):
            # Checking if the fitness exists prevents setting to 0 a fitness that has been initialized by another init
            # call.
            fit = self.fitness
        else:
            fit = PeculiarFitness(n_objectives=n_objectives)
        ConfidentIndividual.__init__(self, fitness=fit)
        FitIndividualByListlike.__init__(self, fitness=fit, seq=seq)
        self.__stats = {}  # dictionary of key -> stat_value
        self.__predictors = [None]*n_objectives
        self.__personalized_feature_importance = None

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

    def set_predictors(self, predictors: Sequence[Predictor]):
        if len(predictors) != self.n_objectives():
            raise ValueError()
        self.__predictors = check_none(predictors)

    def get_predictors(self) -> Sequence[Predictor]:
        return self.__predictors

    def reset_predictors(self):
        self.__predictors = [None]*self.n_objectives()

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

    def __str__(self) -> str:
        res = FitIndividualByListlike.__str__(self)
        res += "Standard deviations: " + iterable_to_string(li=self.std_dev()) + "\n"
        res += "Confidence intervals: " + iterable_to_string(li=self.ci95()) + "\n"
        res += "Bootstrap means: " + iterable_to_string(li=self.bootstrap_mean()) + "\n"
        if self.has_personalized_feature_importance():
            res += "Personalized feature importance: " + str(self.get_personalized_feature_importance()) + "\n"
        else:
            res += "No personalized feature importance.\n"
        return res

    def mothball(self) -> PeculiarIndividualByListlike:
        res = deepcopy(self)
        res.reset_stats()
        res.reset_personalized_feature_importance()
        res.reset_predictors()
        return res
