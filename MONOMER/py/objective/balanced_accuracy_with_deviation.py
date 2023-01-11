import statistics

from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor
from objective.objective_computer import ClassificationObjectiveComputer, BalancedAccuracy


DEFAULT_MAX_DEVIATION = 0.05


class WithDeviationWrapper(ClassificationObjectiveComputer):
    __inner: ClassificationObjectiveComputer
    __max_sd: float

    def __init__(self, inner: ClassificationObjectiveComputer, max_sd: float = DEFAULT_MAX_DEVIATION):
        self.__inner = inner
        self.__max_sd = max_sd

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test, y_test) -> float:
        return self.__inner.compute_from_predictor_and_test(predictor=predictor, x_test=x_test, y_test=y_test)

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

    def _combine_fold_results(self, fold_results: [float]) -> float:
        from_inner = self.__inner._combine_fold_results(fold_results=fold_results)
        sd = statistics.stdev(fold_results)
        if sd > self.__max_sd:
            return 0.0
        else:
            return from_inner

    def compute_from_classes(
            self, hyperparams, hp_manager: HyperparamManager,
            test_pred, test_true, train_pred=None, train_true=None) -> float:
        return self.__inner.compute_from_classes(hyperparams=hyperparams, hp_manager=hp_manager,
                                                 test_pred=test_pred, test_true=test_true,
                                                 train_pred=train_pred, train_true=train_true)


class BalancedAccuracyWithDeviation(WithDeviationWrapper):

    def __init__(self, max_sd: float = DEFAULT_MAX_DEVIATION):
        WithDeviationWrapper.__init__(self, inner=BalancedAccuracy(), max_sd=max_sd)
