from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union, Optional

from pandas import DataFrame

from bootstrap.bootstrap_distribution import bootstrap_ci95_from_classes
from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor, Model
from objective.objective_with_importance.objective_computer_with_importance import \
    DEFAULT_N_RESAMPLES, ObjectiveComputerWithImportance
from objective.social_objective import SocialObjective
from util.hyperbox.hyperbox import ConcreteInterval
from util.utils import IllegalStateError


class SocialObjectiveWithImportance(SocialObjective, ABC):
    """An objective that can depend on the current non-dominated front. Note that an objective that does not depend
    on social information is a special case."""

    def compute_from_classes_with_confidence(
            self, hyperparams, hp_manager: HyperparamManager, test_pred, test_true,
            train_pred=None, train_true=None, compute_confidence: bool = False) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists.
        The nature of an y element is problem dependent and can also be censored data.
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
        cv_result = objective_computer.compute_from_classes(
            hyperparams=hyperparams, hp_manager=hp_manager,
            test_pred=test_pred, test_true=test_true, train_pred=train_pred, train_true=train_true)
        if compute_confidence:
            if test_pred is None:
                cv_result.set_std_dev(std_dev=0.0)
                fitness = cv_result.fitness()
                cv_result.set_ci95(ci95=ConcreteInterval(a=fitness, b=fitness))
                cv_result.set_bootstrap_mean(bootstrap_mean=fitness)
            else:
                ci95, std_dev, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=objective_computer, pred_y_test=test_pred, true_y_test=test_true,
                    pred_y_train=train_pred, true_y_train=train_true, n_resamples=DEFAULT_N_RESAMPLES)
                cv_result.set_std_dev(std_dev=std_dev)
                cv_result.set_ci95(ci95=ci95)
                cv_result.set_bootstrap_mean(bootstrap_mean=b_mean)
        return cv_result

    def compute_from_hyperparams_with_importance(
            self, hyperparams,
            hp_manager: HyperparamManager = DummyHpManager()) -> CVResult:
        """This method might fail if also predictions are needed."""
        cv_result = self.compute_from_classes(hyperparams=hyperparams, hp_manager=hp_manager,
                                              test_pred=None, test_true=None, train_pred=None, train_true=None)
        # Cannot compute importances not having the input values.
        # [0.0] * hp_manager.n_active_features(hyperparams=hyperparams))
        return cv_result

    @abstractmethod
    def objective_computer(self) -> ObjectiveComputerWithImportance:
        raise NotImplementedError()

    def compute_from_predictor_and_test_with_importance(
            self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
            x_train: Union[DataFrame, None] = None,
            y_train: Union[DataFrame, None] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        return self.objective_computer().compute_from_predictor_and_test_with_importance(
            predictor=predictor,
            x_test=x_test, y_test=y_test,
            x_train=x_train, y_train=y_train,
            compute_fi=compute_fi,
            compute_confidence=compute_confidence)

    def compute_from_predictor_and_test_with_importance_all(
            self, predictors: Sequence[Predictor], x_test: DataFrame, y_test: DataFrame,
            x_train: Union[DataFrame, None] = None,
            y_train: Union[DataFrame, None] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> Sequence[CVResult]:
        """Faster than calling iteratively for each predictor because the bootstrapped pools are created
        only once for all the predictors."""
        return self.objective_computer().compute_from_predictor_and_test_with_importance_all(
            predictors=predictors,
            x_test=x_test, y_test=y_test,
            x_train=x_train, y_train=y_train,
            compute_fi=compute_fi,
            compute_confidence=compute_confidence)

    def change_computer(self, objective_computer: ObjectiveComputerWithImportance
                        ) -> SocialObjectiveWithImportance:
        """Returns a new instance."""
        model = None
        if self.has_model():
            model = self.model()
        outcome_label = None
        if self.has_outcome_label():
            outcome_label = self.outcome_label()
        return CompositeSocialObjectiveWithImportance(
            objective_computer=objective_computer,
            model=model,
            target_label=outcome_label)


class CompositeSocialObjectiveWithImportance(SocialObjectiveWithImportance):

    __objective_computer: ObjectiveComputerWithImportance
    __model: Optional[Model]
    __target_label: Optional[str]

    def __init__(self, objective_computer: ObjectiveComputerWithImportance,
                 model: Optional[Model] = None, target_label: Optional[str] = None):
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

    def model(self) -> Model:
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
