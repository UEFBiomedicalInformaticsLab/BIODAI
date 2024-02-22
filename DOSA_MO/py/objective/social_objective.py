from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Union, Optional

from pandas import DataFrame

from hyperparam_manager.dummy_hp_manager import DummyHpManager
from hyperparam_manager.hyperparam_manager import HyperparamManager

from model.model import ClassModel, Predictor
from objective.objective_computer import ObjectiveComputer
from util.named import NickNamed
from util.utils import IllegalStateError
from cross_validation.single_objective.cv_result import CVResult
from util.hyperbox.hyperbox import Interval


def one_objective_nick(computer_nick: str, inner_model_nick: Optional[str] = None) -> str:
    if inner_model_nick is not None:
        return inner_model_nick + "_" + computer_nick
    else:
        return computer_nick


class SocialObjective(NickNamed, ABC):
    """An objective that can depend on the current non-dominated front. Note that an objective that does not depend
    on social information is a special case."""

    def compute_from_classes(
            self, hyperparams, hp_manager: HyperparamManager, test_pred, test_true,
            train_pred=None, train_true=None) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data."""
        if not self.is_class_based():
            raise IllegalStateError("This objective does not work with classifications:\n" + str(self))
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
        return self.objective_computer().compute_from_classes(
            hyperparams=hyperparams, hp_manager=hp_manager,
            test_pred=test_pred, test_true=test_true, train_pred=train_pred, train_true=train_true)

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test, y_test,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        """This method might fail if also predictions on training set are needed but not passed,
        or if the hyperparameters are needed."""
        return self.objective_computer().compute_from_predictor_and_test(
            predictor=predictor, x_test=x_test, y_test=y_test)

    def compute_from_predictor_and_test_all(self, predictors: Sequence[Predictor], x_test, y_test,
                                            x_train: Union[DataFrame, None] = None,
                                            y_train: Union[DataFrame, None] = None) -> Sequence[CVResult]:
        """This method might fail if also predictions on training set are needed but not passed,
        or if the hyperparameters are needed."""
        return [self.compute_from_predictor_and_test(
            predictor=p, x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train) for p in predictors]

    def compute_from_hyperparams(self, hyperparams,
                                 hp_manager: HyperparamManager = DummyHpManager()) -> CVResult:
        """This method might fail if also predictions are needed.
        No feature is predictive according to this objective, and we assign 0 importance to everyone.
        Also confidence is returned because it will be fast to compute: no data is used."""
        return self.objective_computer().compute_from_structure_with_importance(
            hyperparams=hyperparams, hp_manager=hp_manager,
            x=None, y=None, compute_confidence=True)

    def compute_from_hyperparams_all(self, hyperparams_seq: Sequence,
                                     hp_manager: HyperparamManager = DummyHpManager()) -> Sequence[CVResult]:
        return [self.compute_from_hyperparams(hyperparams=h, hp_manager=hp_manager) for h in hyperparams_seq]

    @abstractmethod
    def objective_computer(self) -> ObjectiveComputer:
        raise NotImplementedError()

    def is_class_based(self) -> bool:
        return self.objective_computer().is_class_objective_computer()

    def is_survival(self) -> bool:
        return self.objective_computer().is_survival_objective_computer()

    def is_structural(self):
        return self.objective_computer().is_structural_objective_computer()

    def update(self, hp_pop):
        pass

    def requires_predictions(self) -> bool:
        """False if predictions are not needed."""
        return self.objective_computer().requires_predictions()

    # If true requires an update at every generation.
    @staticmethod
    def is_dynamic():
        return True

    def requires_training_predictions(self) -> bool:
        """
        If true requires that training predictions and actual values are passed. Otherwise it works also without them.
        An objective that works without training data can be computed on training data using it in place of the
        testing data.
            """
        return self.objective_computer().requires_training_predictions()

    def has_model(self) -> bool:
        return False

    def has_outcome_label(self) -> bool:
        return False

    def model(self) -> ClassModel:
        raise IllegalStateError()

    def outcome_label(self) -> str:
        raise IllegalStateError()

    def val_to_label(self, value: Union[float, Interval]) -> Union[float, Interval]:
        """
        Value of the objective to pretty value used in plots, logs, etc. Defaults to identity function.
            """
        return self.objective_computer().val_to_label(value=value)

    def vals_to_labels(self, values: Iterable[Union[float, Interval]]) -> list[Union[float, Interval]]:
        return self.objective_computer().vals_to_labels(values=values)

    def computer_nick(self) -> str:
        return self.objective_computer().nick()

    def computer_name(self) -> str:
        return self.objective_computer().name()

    def nick(self) -> str:
        if self.has_model():
            return one_objective_nick(computer_nick=self.computer_nick(), inner_model_nick=self.model().nick())
            # Subclasses will benefit from this.
        else:
            return one_objective_nick(computer_nick=self.computer_nick())

    def name(self):
        """As outcome key for InputData use .outcome_label() instead."""
        if self.has_model():
            return self.outcome_label() + " " + self.model().name() + " " + self.computer_name()
            # Subclasses will benefit from this.
        else:
            return self.computer_name()

    def __str__(self):
        if self.has_model():
            return self.outcome_label() + " " + str(self.model()) + " " + str(self.objective_computer())
            # Subclasses will benefit from this.
        else:
            return str(self.objective_computer())


class PersonalObjective(SocialObjective, ABC):

    @staticmethod
    def is_dynamic():
        return False
