from abc import ABC, abstractmethod

from pandas import DataFrame

from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Model
from model.survival_model import CoxPredictor, SurvivalPredictor
from objective.objective_computer import ObjectiveComputer
from util.utils import IllegalStateError


class SurvivalObjectiveComputer(ObjectiveComputer, ABC):

    @abstractmethod
    def compute_from_predictor_and_test(self, predictor: SurvivalPredictor, x_test, y_test) -> float:
        raise NotImplementedError()

    def is_class_objective_computer(self) -> bool:
        return False

    def is_survival_objective_computer(self) -> bool:
        return True

    @staticmethod
    def requires_training_predictions() -> bool:
        return False

    def compute_from_classes(
            self, hyperparams, hp_manager: HyperparamManager,
            test_pred, test_true, train_pred=None, train_true=None) -> float:
        raise IllegalStateError()

    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> float:
        raise IllegalStateError()


class CIndex(SurvivalObjectiveComputer):

    def compute_from_predictor_and_test(self, predictor: CoxPredictor, x_test, y_test) -> float:
        return predictor.score_concordance_index(x_test=x_test, y_test=y_test)

    def nick(self) -> str:
        return "c-index"
