from __future__ import annotations
from abc import ABC
from typing import Optional

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.confident_individual import ConfidentIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from individual.predictive_individual import PredictiveIndividualSV, PredictiveIndividual, PredictiveIndividualMV


class ConfidentPredictiveIndividual(ConfidentIndividual, PredictiveIndividual, ABC):
    pass


class ConfidentPredictiveIndividualSV(ConfidentPredictiveIndividual, PredictiveIndividualSV, ABC):
    pass


class ConfidentPredictiveIndividualMV(ConfidentPredictiveIndividual, PredictiveIndividualMV, ABC):
    pass

class ConfidentPredictiveIndividualSparseMV(ConfidentPredictiveIndividualMV, PeculiarIndividualSparse):

    def __init__(self, fitness: HighBestFitness, seq=()):
        """Passed seq is safe-copied."""
        ConfidentPredictiveIndividualMV.__init__(self=self, fitness=fitness)
        PeculiarIndividualSparse.__init__(self=self, n_objectives=fitness.n_objectives(), seq=seq)

    @staticmethod
    def create_from_individual(
            individual: ConfidentPredictiveIndividualMV,
            hp_manager: Optional[MvHyperparamManager] = None) -> ConfidentPredictiveIndividualMV:
        """If an hp manager is not provided, the new individual is created with a sequence of hyperparams that
        is the same sequence of the passed individual. I.e. it is assumed to be codified in the same way.
        If an hp manager is provided, the individual is codified as a mask on all the used features. So not
        a mask on just the predictive ones."""
        seq = individual
        if hp_manager is not None:
            seq = hp_manager.collapsed_used_features_mask(hyperparams=individual)
        res = ConfidentPredictiveIndividualSparseMV(fitness=individual.get_test_fitness(), seq=seq)
        res.set_ci95(individual.ci95())
        res.set_std_dev(individual.std_dev())
        res.set_bootstrap_mean(individual.bootstrap_mean())
        res.set_predictors(individual.get_predictors())
        if isinstance(individual, PeculiarIndividual):
            res.set_crowding_distance(individual.get_crowding_distance())
            res.set_peculiarity(individual.get_peculiarity())
            res.set_social_space(individual.get_social_space())
        if isinstance(individual, PeculiarIndividualByListlike):
            res.set_stats(individual.get_stats())
            if individual.has_personalized_feature_importance():
                res.set_personalized_feature_importance(individual.get_personalized_feature_importance())
        return res

