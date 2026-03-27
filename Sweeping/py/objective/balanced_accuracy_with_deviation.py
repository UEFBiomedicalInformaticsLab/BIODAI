import statistics
from collections.abc import Sequence
from typing import Optional, Any

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from objective.objective_computer import CrispClassificationObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from util.hyperbox.hyperbox import ConcreteInterval

DEFAULT_MAX_DEVIATION = 0.05


class WithDeviationWrapper(CrispClassificationObjectiveComputer):
    __inner: CrispClassificationObjectiveComputer
    __max_sd: float

    def __init__(self, inner: CrispClassificationObjectiveComputer, max_sd: float = DEFAULT_MAX_DEVIATION):
        self.__inner = inner
        self.__max_sd = max_sd

    def base_nick(self) -> str:
        return self.__inner.nick() + "_sd"

    def nick(self) -> str:
        return self.base_nick() + str(self.__max_sd)

    def name(self) -> str:
        return self.__inner.name() + " with max sd " + str(self.__max_sd)

    def __str__(self) -> str:
        return str(self.__inner) + " with max standard deviation " + str(self.__max_sd)

    @staticmethod
    def requires_predictions():
        return True

    def force_general_cv(self) -> bool:
        """Return true to force the use of general cv when classification cv would be used otherwise."""
        return True

    def _combine_fold_results(self, fold_results: Sequence[CVResult]) -> CVResult:
        from_inner = self.__inner._combine_fold_results(fold_results=fold_results)
        sd = statistics.stdev([r.fitness() for r in fold_results])  # standard deviation between folds.
        if sd > self.__max_sd:
            return CVResult(fitness=0.0, std_dev=0.0, ci95=ConcreteInterval(0.0,0.0))
        else:
            return from_inner

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None
                                ) -> CVResult:
        return self.__inner.compute_from_classes_mv(test_pred=test_pred, test_true=test_true, train_pred=train_pred,
                                                    train_true=train_true, hyperparams=hyperparams,
                                                    hp_manager=hp_manager)


class BalancedAccuracyWithDeviation(WithDeviationWrapper):

    def __init__(self, max_sd: float = DEFAULT_MAX_DEVIATION):
        WithDeviationWrapper.__init__(self, inner=BalancedAccuracy(), max_sd=max_sd)
