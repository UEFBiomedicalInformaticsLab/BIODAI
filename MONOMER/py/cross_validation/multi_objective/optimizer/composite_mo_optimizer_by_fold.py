from cross_validation.multi_objective.optimizer.mo_optimizer_including_feature_importance import \
    MOOptimizerIncludingFeatureImportance, nick_from_optimizer_and_fi, name_from_optimizer_and_fi
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold
from feature_importance.feature_importance_by_fold import FeatureImportanceByFold
from util.utils import name_value


class CompositeMultiObjectiveOptimizerByFold(MultiObjectiveOptimizerByFold):
    """Creates feature importance specific for fold and pipes it to optimizer."""
    __fi: FeatureImportanceByFold
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance

    def __init__(self,
                 fi: FeatureImportanceByFold,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance):
        self.__fi = fi
        self.__optimizer = optimizer

    def optimizer_for_fold(self, fold_index: int) -> MultiObjectiveOptimizer:
        return MOOptimizerIncludingFeatureImportance(
            feature_importance=self.__fi.fi_for_fold(fold_index=fold_index),
            optimizer=self.__optimizer)

    def optimizer_for_all_data(self) -> MultiObjectiveOptimizer:
        return MOOptimizerIncludingFeatureImportance(
            feature_importance=self.__fi.fi_for_all_data(),
            optimizer=self.__optimizer)

    def uses_inner_models(self) -> bool:
        return self.__optimizer.uses_inner_models()

    def __str__(self) -> str:
        res = "Multi-objective optimizer by fold\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        res += "Multi-view feature importance strategy by fold:\n"
        res += str(self.__fi) + "\n"
        return res

    def nick(self) -> str:
        return nick_from_optimizer_and_fi(optimizer=self.__optimizer, fi=self.__fi)

    def name(self) -> str:
        return name_from_optimizer_and_fi(optimizer=self.__optimizer, fi=self.__fi)
