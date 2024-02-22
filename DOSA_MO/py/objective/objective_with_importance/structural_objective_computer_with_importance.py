from abc import ABC
from typing import Union

from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Model, Predictor
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from util.utils import IllegalStateError


class StructuralObjectiveComputerWithImportance(ObjectiveComputerWithImportance, ABC):

    def _compute_with_kfold_cv_class_with_importance(self, model: Model, x: DataFrame, y: DataFrame, folds_list,
                                                     compute_fi: bool = False,
                                                     compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        raise IllegalStateError()

    def is_class_objective_computer(self) -> bool:
        return False

    def is_survival_objective_computer(self) -> bool:
        return False

    def is_structural_objective_computer(self) -> bool:
        return True

    @staticmethod
    def requires_training_predictions() -> bool:
        return False

    def requires_predictions(self) -> bool:
        return False

    def can_compute_from_classes(self) -> bool:
        return False

    def compute_from_classes(self, hyperparams, hp_manager: Union[HyperparamManager, None], test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        raise IllegalStateError()

    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        raise IllegalStateError()
