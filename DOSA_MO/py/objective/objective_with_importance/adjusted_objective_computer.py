from abc import ABC
from typing import Union

from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from fitness_adjuster.fitness_adjuster import FitnessAdjuster
from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor, Model
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance, \
    ClassificationObjectiveComputerWithImportance
from objective.objective_with_importance.structural_objective_computer_with_importance import \
    StructuralObjectiveComputerWithImportance
from objective.objective_with_importance.survival_objective_computer_with_importance import \
    SurvivalObjectiveComputerWithImportance
from util.dataframes import n_col


def adjusted_nick(inner: str) -> str:
    return "adj_" + inner


class AdjustedObjectiveComputer(ObjectiveComputerWithImportance, ABC):
    __inner: ObjectiveComputerWithImportance
    __adjuster: FitnessAdjuster

    def __init__(self,
                 inner: ObjectiveComputerWithImportance,
                 adjuster: FitnessAdjuster):
        self.__inner = inner
        self.__adjuster = adjuster

    def compute_from_classes(self, hyperparams, hp_manager: Union[HyperparamManager, None], test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        inner_res = self.__inner.compute_from_classes(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            test_pred=test_pred,
            test_true=test_true,
            train_pred=train_pred,
            train_true=train_true)
        n_features = hp_manager.n_active_features(hyperparams)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        """x includes only the features to actually use.
        This method might fail if also training data is needed but not passed,
        or if the hyperparameters are needed."""
        inner_res = self.__inner.compute_from_predictor_and_test(
            predictor=predictor,
            x_test=x_test,
            y_test=y_test,
            x_train=x_train,
            y_train=y_train)
        n_features = n_col(x_test)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def name(self) -> str:
        return "adjusted " + self.__inner.name()

    def nick(self) -> str:
        return adjusted_nick(self.__inner.nick())

    def __compute_from_inner_res(self, inner_res: CVResult, num_features: int) -> CVResult:
        if inner_res.has_std_dev():
            fai = FitnessAdjusterInput(
                original_fitness=inner_res.fitness(),
                std_dev=inner_res.std_dev(),
                num_features=num_features,
                bootstrap_mean=inner_res.bootstrap_mean())
            adj_fitness = self.__adjuster.adjust_fitness(input_data=fai)
            res = CVResult(fitness=adj_fitness)
            if inner_res.has_importances():
                res.set_importances(inner_res.importances())
            if inner_res.has_final_predictor():
                res.set_final_predictor(inner_res.final_predictor())
            return res
        else:
            raise ValueError("Standard deviation is needed to adjust the fitness.")

    def _compute_with_kfold_cv_class_with_importance(self, model: Model, x, y: DataFrame, folds_list,
                                                     compute_fi: bool = False,
                                                     compute_confidence: bool = False) -> CVResult:
        """x and y need to be already selected.
        Inner results include confidence in any case because it is needed for the adjustment."""
        inner_res = self.__inner._compute_with_kfold_cv_class_with_importance(
            model=model,
            x=x,
            y=y,
            folds_list=folds_list,
            compute_fi=compute_fi,
            compute_confidence=True)
        n_features = n_col(x)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        """x and y need to be already selected. x already filtered by columns if necessary."""
        inner_res = self.__inner._compute_with_kfold_cv_class(
            model=model,
            x=x,
            y=y,
            folds_list=folds_list)
        n_features = n_col(x)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def requires_target(self) -> bool:
        return self.__inner.requires_target()

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        inner_res = self.__inner.compute_from_structure_with_importance(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            x=x,
            y=y,
            compute_fi=False,
            compute_confidence=True)
        n_features = hp_manager.n_active_features(hyperparams)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def _compute_with_kfold_cv_general_with_importance(
            self, model: Model, x, y, folds_list,
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x includes only the features to actually use."""
        inner_res = self.__inner._compute_with_kfold_cv_general_with_importance(
            model=model,
            x=x,
            y=y,
            folds_list=folds_list,
            compute_fi=compute_fi,
            compute_confidence=True)
        n_features = n_col(x)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)


class ClassificationAdjustedObjectiveComputer(AdjustedObjectiveComputer, ClassificationObjectiveComputerWithImportance):

    def __init__(self,
                 inner: ObjectiveComputerWithImportance,
                 adjuster: FitnessAdjuster):
        AdjustedObjectiveComputer.__init__(self=self, inner=inner, adjuster=adjuster)


class SurvivalAdjustedObjectiveComputer(
        AdjustedObjectiveComputer, SurvivalObjectiveComputerWithImportance):

    def __init__(self,
                 inner: ObjectiveComputerWithImportance,
                 adjuster: FitnessAdjuster):
        AdjustedObjectiveComputer.__init__(self=self, inner=inner, adjuster=adjuster)


class StructuralAdjustedObjectiveComputer(
        AdjustedObjectiveComputer, StructuralObjectiveComputerWithImportance):

    def __init__(self,
                 inner: ObjectiveComputerWithImportance,
                 adjuster: FitnessAdjuster):
        AdjustedObjectiveComputer.__init__(self=self, inner=inner, adjuster=adjuster)
