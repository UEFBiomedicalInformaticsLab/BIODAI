import traceback
from abc import ABC
from typing import Union

import pandas as pd
from numpy import unique
from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Model
from model.survival_model import CoxPredictor
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from util.survival.survival_utils import survival_times, integrated_brier_score_from_df
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

    def compute_from_structure_with_importance(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
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


class NotIBS(SurvivalObjectiveComputerWithImportance):
    __end_of_time: float

    def __init__(self, end_of_time: float):
        self.__end_of_time = end_of_time

    def compute_from_predictor_and_test(self, predictor: CoxPredictor,
                                        x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None,
                                        same_distribution: bool = False) -> CVResult:
        """It is better to use also y_train to learn a censoring predictor, if the distribution is expected to be the
        same.
        same_distribution: set to true if training and testing data are expected to originate from the same
        distribution.
        Time considered goes from 0 to end_of_time."""
        if same_distribution is False:
            print("same_distribution is false")
            traceback.print_stack()
            y_train = y_test.iloc[:0].copy()
        elif y_train is None:
            print("Missing y_train")
            traceback.print_stack()
            y_train = y_test.iloc[:0].copy()
        if y_test is None:
            raise ValueError()
        times = [0.0]
        times.extend(survival_times(y_test))
        times.extend(survival_times(y_train))
        times.append(self.__end_of_time)
        times = unique(times)
        times.sort()
        probs = predictor.predict_survival_probabilities(x_test, times)
        surv_for_censoring = pd.concat([y_train, y_test])
        return CVResult(fitness=1.0 - integrated_brier_score_from_df(
            surv_for_censoring_df=surv_for_censoring, surv_test_df=y_test, estimate=probs.T, times=times))

    def name(self) -> str:
        return "1 - Integrated Brier score"

    def nick(self) -> str:
        return "IBS"
