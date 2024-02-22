from abc import ABC, abstractmethod
from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.composite_mo_optimizer_by_fold import \
    CompositeMultiObjectiveOptimizerByFold
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold, \
    DummyMultiObjectiveOptimizerByFold
from feature_importance.feature_importance_by_fold import FeatureImportanceByFold
from objective.social_objective import PersonalObjective
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMO
from util.named import NickNamed


def create_mo_optimizer_by_fold(
        mo_optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
        feature_importance: FeatureImportanceByFold,
        feature_selector: FeatureSelectorMO) -> MultiObjectiveOptimizerByFold:
    if isinstance(mo_optimizer, MultiObjectiveOptimizerAcceptingFeatureImportance):
        return CompositeMultiObjectiveOptimizerByFold(
            fi_by_fold=feature_importance,
            optimizer=mo_optimizer,
            feature_selector=feature_selector)
    else:
        return DummyMultiObjectiveOptimizerByFold(optimizer=mo_optimizer)


class MOOptimizerFactory(NickNamed, ABC):

    @abstractmethod
    def create_optimizer(
            self,
            objectives: Sequence[PersonalObjective]) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
        raise NotImplementedError()

    def create_optimizer_by_fold(
            self,
            objectives: Sequence[PersonalObjective],
            feature_importance_by_fold: FeatureImportanceByFold,
            feature_selector_mo: FeatureSelectorMO) -> MultiObjectiveOptimizerByFold:
        optimizer = self.create_optimizer(objectives=objectives)
        return create_mo_optimizer_by_fold(
            mo_optimizer=optimizer,
            feature_importance=feature_importance_by_fold,
            feature_selector=feature_selector_mo
        )

    def create_dummy_optimizer_by_fold(
            self,
            objectives: Sequence[PersonalObjective]) -> MultiObjectiveOptimizerByFold:
        optimizer = self.create_optimizer(objectives=objectives)
        return DummyMultiObjectiveOptimizerByFold(optimizer=optimizer)

    @abstractmethod
    def uses_inner_models(self) -> bool:
        raise NotImplementedError()
