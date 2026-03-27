from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import numpy as np

from bootstrap.bootstrap_distribution import bootstrap_ci95_from_classes, DEFAULT_RESAMPLING_SEED
from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.objective_computer_with_importance import \
    DEFAULT_N_RESAMPLES, ObjectiveComputerWithImportance
from objective.social_objective import SocialObjective
from util.hyperbox.hyperbox import ConcreteInterval
from util.randoms import random_seed
from util.utils import IllegalStateError


class SocialObjectiveWithImportance(SocialObjective, ABC):
    """An objective that can depend on the current non-dominated front. Note that an objective that does not depend
    on social information is a special case."""

    def compute_from_classes_with_confidence(
            self, hyperparams, hp_manager: MvHyperparamManager, test_pred: Optional[Sequence], test_true,
            train_pred=None, train_true=None, compute_confidence: bool = False) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists.
        The nature of a y element is problem dependent and can also be censored data.
        Cannot compute importances not having the input values: importances are all zero."""
        if not self.is_class_based():
            raise IllegalStateError("This objective does not work with classifications.")
        if test_pred is not None:
            if len(test_true) != len(test_pred):
                raise ValueError(
                    "len(test_true): " + str(len(test_true)) + " len(test_pred): " + str(len(test_pred)) + "\n" +
                    "test_true:\n" + str(test_true) + "\ntest_pred:\n" + str(test_pred) + "\n")
        if train_pred is not None:
            if len(train_true) != len(train_pred):
                raise ValueError(
                    "len(train_true): " + str(len(train_true)) + " len(train_pred): " + str(len(train_pred)) + "\n" +
                    "train_true:\n" + str(train_true) + "\ntrain_pred:\n" + str(train_pred) + "\n")
        objective_computer = self.objective_computer()
        cv_result = objective_computer.compute_from_classes_mv(
            test_pred=test_pred, test_true=test_true, train_pred=train_pred, train_true=train_true,
            hyperparams=hyperparams, hp_manager=hp_manager)
        if compute_confidence:
            if test_pred is None:
                cv_result.set_std_dev(std_dev=0.0)
                fitness = cv_result.fitness()
                cv_result.set_ci95(ci95=ConcreteInterval(a=fitness, b=fitness))
                cv_result.set_bootstrap_mean(bootstrap_mean=fitness)
            else:
                ci95, std_dev, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=objective_computer, pred_y_test=test_pred, true_y_test=test_true,
                    pred_y_train=train_pred, true_y_train=train_true, n_resamples=DEFAULT_N_RESAMPLES,
                    random=np.random.default_rng(seed=random_seed()))
                cv_result.set_std_dev(std_dev=std_dev)
                cv_result.set_ci95(ci95=ci95)
                cv_result.set_bootstrap_mean(bootstrap_mean=b_mean)
        return cv_result

    @abstractmethod
    def objective_computer(self) -> ObjectiveComputerWithImportance:
        raise NotImplementedError()

    def compute_from_predictor_and_test_with_importance(
            self, predictor: MVPredictor,
            test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        return self.objective_computer().compute_from_predictor_and_test_with_importance(
            predictor=predictor,
            test_data=test_data,
            train_data=train_data,
            compute_fi=compute_fi,
            compute_confidence=compute_confidence)

    def compute_from_predictor_and_test_with_importance_all_mv(
            self, predictors: Sequence[MVPredictor],
            test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None,
            compute_confidence: bool = False, seed: int = DEFAULT_RESAMPLING_SEED) -> Sequence[CVResult]:
        """Faster than calling iteratively for each predictor because the bootstrapped pools might be created
        only once for all the predictors."""
        return self.objective_computer().compute_from_predictor_and_test_with_importance_all_mv(
            predictors=predictors,
            test_data=test_data,
            train_data=train_data,
            compute_confidence=compute_confidence, seed=seed)

    def change_computer(self, objective_computer: ObjectiveComputerWithImportance
                        ) -> SocialObjectiveWithImportance:
        """Returns a new instance."""
        model = None
        if self.has_model():
            model = self.mv_model()
        outcome_label = None
        if self.has_outcome_label():
            outcome_label = self.outcome_label()
        return CompositeSocialObjectiveWithImportance(
            objective_computer=objective_computer,
            model=model,
            target_label=outcome_label)


class CompositeSocialObjectiveWithImportance(SocialObjectiveWithImportance):
    __objective_computer: ObjectiveComputerWithImportance
    __model: Optional[MVModel]
    __target_label: Optional[str]

    def __init__(self, objective_computer: ObjectiveComputerWithImportance,
                 model: Optional[MVModel] = None, target_label: Optional[str] = None):
        self.__objective_computer = objective_computer
        if objective_computer.requires_target():
            if target_label is None and model is not None:
                raise ValueError("If there is a model there must be also a target label.")
            self.__model = model
            self.__target_label = target_label
        else:
            self.__model = None
            self.__target_label = None

    def objective_computer(self) -> ObjectiveComputerWithImportance:
        return self.__objective_computer

    def has_model(self) -> bool:
        return self.__model is not None

    def has_outcome_label(self) -> bool:
        return self.__target_label is not None

    def mv_model(self) -> MVModel:
        if self.has_model():
            return self.__model
        else:
            raise IllegalStateError(str(self))

    def outcome_label(self) -> str:
        """An objective can have no model (e.g. if a single independent model is used for all the objectives)
        but still have an outcome."""
        if self.has_outcome_label():
            return self.__target_label
        else:
            raise IllegalStateError(str(self))
