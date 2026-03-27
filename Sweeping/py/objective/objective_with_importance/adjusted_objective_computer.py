from typing import Optional, Any

from cross_validation.single_objective.cv_result import CVResult
from fitness_adjuster.fitness_adjuster import FitnessAdjuster
from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance


def adjusted_nick(inner: str) -> str:
    return "adj_" + inner


class AdjustedObjectiveComputer(ObjectiveComputerWithImportance):
    __inner: ObjectiveComputerWithImportance
    __adjuster: FitnessAdjuster

    def __init__(self,
                 inner: ObjectiveComputerWithImportance,
                 adjuster: FitnessAdjuster):
        self.__inner = inner
        self.__adjuster = adjuster

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """Uses hp and hp manager to compute the adjustment. Currently, it considers the sum of used features
        (predictive + adjusting)."""
        inner_res = self.__inner.compute_from_classes_mv(test_pred=test_pred, test_true=test_true,
                                                         train_pred=train_pred, train_true=train_true,
                                                         hyperparams=hyperparams, hp_manager=hp_manager)
        n_features = hp_manager.n_used_features(hyperparams)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def compute_from_predictor_and_test_mv(
            self, predictor: MVPredictor, test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None) -> CVResult:
        """x must include only the features to actually use."""
        inner_res = self.__inner.compute_from_predictor_and_test_mv(
            predictor=predictor,
            test_data=test_data,
            train_data=train_data)
        n_features = test_data.n_features()
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

    def _compute_with_kfold_cv_class_with_importance_mv(self, model: MVModel, data: ModelReadyInputData,
                                                        folds_list: list[tuple[IndexArray, IndexArray]],
                                                        compute_fi: bool = False,
                                                        compute_confidence: bool = False) -> CVResult:
        """Features and outcome need to be already selected.
        Inner results include confidence in any case because it is needed for the adjustment."""
        inner_res = self.__inner._compute_with_kfold_cv_class_with_importance_mv(
            model=model,
            data=data,
            folds_list=folds_list,
            compute_fi=compute_fi,
            compute_confidence=True)
        n_features = data.n_features()
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def requires_target(self) -> bool:
        return self.__inner.requires_target()

    def requires_predictions(self) -> bool:
        return self.__inner.requires_predictions()

    def compute_from_structure(
            self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
            data: Optional[ModelReadyInputData] = None) -> CVResult:
        inner_res = self.__inner.compute_from_structure_with_importance(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            data=data,
            compute_fi=False,
            compute_confidence=True)
        n_features = hp_manager.n_used_features(hyperparams)
        return self.__compute_from_inner_res(inner_res=inner_res, num_features=n_features)

    def is_crisp_class_objective_computer(self) -> bool:
        return self.__inner.is_crisp_class_objective_computer()

    def is_proba_class_objective_computer(self) -> bool:
        return self.__inner.is_proba_class_objective_computer()

    def is_survival_objective_computer(self) -> bool:
        return self.__inner.is_survival_objective_computer()

    def is_structural_objective_computer(self) -> bool:
        return self.__inner.is_structural_objective_computer()

    def requires_training_predictions(self) -> bool:
        return self.__inner.requires_training_predictions()

    def can_compute_from_classes(self) -> bool:
        return self.__inner.can_compute_from_classes()
