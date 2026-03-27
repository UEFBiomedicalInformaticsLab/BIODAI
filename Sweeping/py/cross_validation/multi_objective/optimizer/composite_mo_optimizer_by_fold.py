from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold
from cross_validation.multi_objective.optimizer.optimizer_names import nick_from_optimizer_and_fi_and_fs, \
    name_from_optimizer_and_fi_and_fs
from cross_validation.multi_objective.optimizer.prefiltered_mo_optimizer_including_fi import \
    PrefilteredMOOptimizerIncludingFI, ActiveFeaturesObserver
from feature_importance.feature_importance_by_fold import FeatureImportanceByFold
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMO
from util.str_utils import name_value


class CompositeMultiObjectiveOptimizerByFold(MultiObjectiveOptimizerByFold):
    """Creates feature importance specific for fold and pipes it to optimizer."""
    __fi: FeatureImportanceByFold
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __feature_selector: FeatureSelectorMO

    def __init__(self,
                 fi_by_fold: FeatureImportanceByFold,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                 feature_selector: FeatureSelectorMO):
        self.__fi = fi_by_fold
        self.__optimizer = optimizer
        self.__feature_selector = feature_selector

    def optimizer_for_fold(self, fold_index: int) -> MultiObjectiveOptimizer:
        return PrefilteredMOOptimizerIncludingFI(
            feature_importance=self.__fi.fi_for_fold(fold_index=fold_index),
            optimizer=self.__optimizer,
            feature_selector=self.__feature_selector)

    def optimizer_for_all_data(
            self, active_feature_observers: Sequence[ActiveFeaturesObserver] = ()) -> MultiObjectiveOptimizer:
        res = PrefilteredMOOptimizerIncludingFI(
            feature_importance=self.__fi.fi_for_all_data(),
            optimizer=self.__optimizer,
            feature_selector=self.__feature_selector)
        for o in active_feature_observers:
            res.add_active_feature_observer(o)
        return res

    def uses_inner_models(self) -> bool:
        return self.__optimizer.uses_inner_models()

    def __str__(self) -> str:
        res = "Multi-objective optimizer by fold\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Feature selector:\n"
        res += str(self.__feature_selector)
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        res += "Multi-view feature importance strategy by fold:\n"
        res += str(self.__fi) + "\n"
        return res

    def nick(self) -> str:
        return nick_from_optimizer_and_fi_and_fs(optimizer=self.__optimizer, fi=self.__fi, fs=self.__feature_selector)

    def name(self) -> str:
        return name_from_optimizer_and_fi_and_fs(optimizer=self.__optimizer, fi=self.__fi, fs=self.__feature_selector)
