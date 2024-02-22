from abc import ABC
from typing import Union
from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Model
from model.survival_model import CoxPredictor
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from util.utils import IllegalStateError


class SurvivalObjectiveComputer(ObjectiveComputer, ABC):

    def is_class_objective_computer(self) -> bool:
        return False

    def is_survival_objective_computer(self) -> bool:
        return True

    @staticmethod
    def requires_training_predictions() -> bool:
        return False

    def can_compute_from_classes(self) -> bool:
        return False

    @staticmethod
    def requires_target() -> bool:
        return True

    def compute_from_classes(
            self, hyperparams, hp_manager: HyperparamManager,
            test_pred, test_true, train_pred=None, train_true=None) -> CVResult:
        raise IllegalStateError()

    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        raise IllegalStateError()

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        raise IllegalStateError()


class SurvivalObjectiveComputerWithImportance(
        ObjectiveComputerWithImportance, SurvivalObjectiveComputer, ABC):

    def is_structural_objective_computer(self) -> bool:
        return False

    def _compute_with_kfold_cv_class_with_importance(
            self, model: Model, x, y: DataFrame, folds_list,
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()

    def compute_from_structure_with_importance(
            self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
            y: DataFrame, compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()


class CIndex(SurvivalObjectiveComputerWithImportance):

    def compute_from_predictor_and_test(self, predictor: CoxPredictor,
                                        x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        try:
            return CVResult(fitness=predictor.score_concordance_index(x_test, y_test))
            # We do not use parameter keys since they are different for single-view and multi-view
        except BaseException as e:
            raise Exception(str(e) + "\nPredictor: " + str(predictor) + "\nx_test:\n" + str(x_test) + "\n")

    def name(self) -> str:
        return "concordance index"

    def nick(self) -> str:
        return "c-index"
