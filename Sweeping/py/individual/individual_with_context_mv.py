from __future__ import annotations

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV, \
    ConfidentPredictiveIndividualSparseMV
from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.individual_with_context import IndividualWithContext
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.list_like import BoolListLike
from util.preconditions import check_none
from util.utils import IllegalStateError


class IndividualWithContextMV(ConfidentPredictiveIndividualMV, IndividualWithContext):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified.
    An individual with context has an embedded hp manager that
    allows for direct extraction of the used features. The individual, seen
    as a sequence, is a mask of the used features of the collapsed views.
    I.e. the individual is the result of calling collapsed_used_features_mask of the embedded hp manager on
    the nested individual."""
    __hp_manager: MvHyperparamManager

    def __init__(self, individual: PeculiarIndividualByListlike, hp_manager: MvHyperparamManager):
        if not isinstance(individual, PeculiarIndividualByListlike):
            raise IllegalStateError("individual is not a PeculiarIndividualByListlike\n" +
                                    "individual:\n" +
                                    str(individual))
        individual = ConfidentPredictiveIndividualSparseMV.create_from_individual(individual=individual)
        IndividualWithContext.__init__(self=self, individual=individual)
        self.__hp_manager = check_none(hp_manager)

    def used_features_masks(self) -> dict[str, BoolListLike]:
        return self.__hp_manager.used_feature_masks(hyperparams=self._individual)

    def _hp_manager(self) -> MvHyperparamManager:
        return self.__hp_manager

    def downlift(self, lifter: FeatureSpaceLifterMV) -> IndividualWithContextMV:
        predictive_individual = self._individual
        downlifted_predictive_individual = ConfidentPredictiveIndividualSparseMV(
                fitness=PeculiarFitness.create_from_high_best_fitness(predictive_individual.get_test_fitness()),
                seq=lifter.collapse().downlift(features=self.collapsed_used_features_mask()))
        downlifted_predictive_individual.set_std_dev(predictive_individual.std_dev())
        downlifted_predictive_individual.set_ci95(predictive_individual.ci95())
        downlifted_predictive_individual.set_bootstrap_mean(predictive_individual.bootstrap_mean())
        predictors = []
        for p in predictive_individual.get_predictors():
            # We know this method exists since it is a PredictiveIndividual
            if p is None:
                predictors.append(None)
            else:
                predictors.append(p.downlift(lifter=lifter))
        downlifted_predictive_individual.set_predictors(predictors=predictors)
        return IndividualWithContextMV(
            individual=downlifted_predictive_individual,
            hp_manager=lifter.lower_space_dummy_mv_hp_manager())
        # The individual is now a Boolean sequence, so the dummy is fine.

    def __str__(self) -> str:
        base = super().__str__()
        return base + "Hyperparam manager:\n" + str(self.__hp_manager) + "\n"

