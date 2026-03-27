from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Union, Optional, Any

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel

from model.sv_model import SVModel
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from util.named import NickNamed
from util.utils import IllegalStateError
from cross_validation.single_objective.cv_result import CVResult
from util.hyperbox.hyperbox import Interval
from model.multi_view.mv_predictor import MVPredictor


def one_objective_nick(computer_nick: str, inner_model_nick: Optional[str] = None) -> str:
    if inner_model_nick is not None:
        return inner_model_nick + "_" + computer_nick
    else:
        return computer_nick


class SocialObjective(NickNamed, ABC):
    """An objective that can depend on the current non-dominated front. Note that an objective that does not depend
    on social information is a special case. The performance requires an update at least once per generation.
    This is the base class of all the objectives of this program."""

    def compute_from_classes(
            self, test_pred, test_true,
            train_pred=None, train_true=None,
            hyperparams: Optional[Any] = None,
            hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of a y element is problem dependent and can also be censored data.
        Uses hp and hp manager to compute the adjustment when the objective is adjusted for overestimation."""
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
        return self.objective_computer().compute_from_classes_mv(
            test_pred=test_pred, test_true=test_true, train_pred=train_pred, train_true=train_true,
            hyperparams=hyperparams, hp_manager=hp_manager)

    def compute_from_predictor_and_test_mv(self, predictor: MVPredictor,
                                           test_data: ModelReadyInputData,
                                           train_data: Optional[ModelReadyInputData] = None) -> CVResult:
        """This method might fail if also predictions on training set are needed but not passed,
        or if the hyperparameters are needed."""
        return self.objective_computer().compute_from_predictor_and_test_mv(
            predictor=predictor, test_data=test_data, train_data=train_data)

    def compute_from_predictor_and_test_all_mv(self, predictors: Sequence[MVPredictor],
                                               test_data: ModelReadyInputData,
                                               train_data: Optional[ModelReadyInputData] = None) -> Sequence[CVResult]:
        """This method might fail if also predictions on training set are needed but not passed,
        or if the hyperparameters are needed."""
        return [self.compute_from_predictor_and_test_mv(
            predictor=p, test_data=test_data, train_data=train_data) for p in predictors]

    def compute_from_hyperparams(self, hyperparams,
                                 hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """This method might fail if also predictions are needed.
        No feature is predictive according to this objective, and we assign 0 importance to everyone.
        Also confidence is returned because it will be fast to compute: no data is used."""
        return self.objective_computer().compute_from_structure_with_importance(
            hyperparams=hyperparams, hp_manager=hp_manager, compute_confidence=True)

    def compute_from_hyperparams_all(self, hyperparams_seq: Sequence,
                                     hp_manager: Optional[MvHyperparamManager] = None) -> Sequence[CVResult]:
        return [self.compute_from_hyperparams(hyperparams=h, hp_manager=hp_manager) for h in hyperparams_seq]

    @abstractmethod
    def objective_computer(self) -> ObjectiveComputerWithImportance:
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

    def sv_model(self) -> SVModel:
        return self.mv_model().as_sv_model()

    @abstractmethod
    def mv_model(self) -> MVModel:
        raise NotImplementedError()

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
            return one_objective_nick(computer_nick=self.computer_nick(), inner_model_nick=self.mv_model().nick())
            # Subclasses will benefit from this.
        else:
            return one_objective_nick(computer_nick=self.computer_nick())

    def name(self):
        """As outcome key for InputData use .outcome_label() instead."""
        if self.has_model():
            return self.outcome_label() + " " + self.mv_model().name() + " " + self.computer_name()
            # Subclasses will benefit from this.
        else:
            return self.computer_name()

    def __str__(self):
        if self.has_model():
            return self.outcome_label() + " " + str(self.mv_model()) + " " + str(self.objective_computer())
            # Subclasses will benefit from this.
        else:
            return str(self.objective_computer())


class PersonalObjective(SocialObjective, ABC):
    pass
