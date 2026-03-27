from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Union, Optional, Any

from cross_validation.single_objective.cv_result import CVResult
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.input_data import InputData
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.math.list_math import vector_mean
from util.math.utils import std_dev_of_uncorrelated_mean, confidence_interval_of_uncorrelated_mean
from util.named import NickNamed
from util.math.summer import KahanSummer
from util.table.table import Table
from util.utils import IllegalStateError


class ObjectiveComputer(NickNamed, ABC):

    @abstractmethod
    def compute_from_predictor_and_test_mv(self, predictor: MVPredictor, test_data: ModelReadyInputData,
                                           train_data: Optional[ModelReadyInputData] = None) -> CVResult:
        """x includes only the features to actually use.
        This method might fail if also training data is needed but not passed,
        or if the hyperparameters are needed (in which case this is not the correct method to call)."""
        raise NotImplementedError()

    def is_class_objective_computer(self) -> bool:
        """Either crisp or proba."""
        return self.is_crisp_class_objective_computer() or self.is_proba_class_objective_computer()

    @abstractmethod
    def is_crisp_class_objective_computer(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_proba_class_objective_computer(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_survival_objective_computer(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_structural_objective_computer(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def requires_predictions() -> bool:
        """Override to return false if predictions are not needed."""
        return True

    @staticmethod
    @abstractmethod
    def requires_target() -> bool:
        """If true requires that a target is defined.
        An objective might require targets (labels) but not predictions."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def requires_training_predictions() -> bool:
        raise NotImplementedError()

    def val_to_label(self, value: Union[float, Interval]) -> Union[float, Interval]:
        """Value of the objective to pretty value used in plots, logs, etc. Defaults to identity function."""
        if isinstance(value, Interval):
            return ConcreteInterval(self.val_to_label_float(value.a()), self.val_to_label_float(value.b()))
        else:
            return self.val_to_label_float(value)

    @staticmethod
    def val_to_label_float(value: float) -> float:
        """Value of the objective to pretty value used in plots, logs, etc. Defaults to identity function."""
        return value

    def vals_to_labels(self, values: Iterable[Union[float, Interval]]) -> list[Union[float, Interval]]:
        return [self.val_to_label(x) for x in values]

    @abstractmethod
    def nick(self) -> str:
        raise NotImplementedError()

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()

    @abstractmethod
    def can_compute_from_classes(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data.
        Sets also feature importances as uniformly 0.
        Throws exception if not applicable for this objective."""
        raise NotImplementedError()

    def force_general_cv(self) -> bool:
        """Return true to force the use of general cv when classification cv would be used otherwise."""
        return False

    @staticmethod
    def _fold_input_data(data: InputData, fold: tuple[IndexArray, IndexArray]) -> tuple[InputData,InputData]:
        train_selection = fold[0]
        test_selection = fold[1]
        train_data = data.select_samples(row_indices=train_selection)
        test_data = data.select_samples(row_indices=test_selection)
        return train_data, test_data

    @staticmethod
    def _fold_data(all_x: Table, all_y, fold: tuple[IndexArray, IndexArray]) -> tuple[Table, Any, Table, Any]:
        train_mask = fold[0]
        test_mask = fold[1]
        x_train = all_x.select_rows(selected=train_mask)
        y_train = all_y.iloc[train_mask]  # TODO This could be a generic Sequence
        x_test = all_x.select_rows(selected=test_mask)
        y_test = all_y.iloc[test_mask]
        return x_train, y_train, x_test, y_test

    def _combine_fold_results(self, fold_results: Sequence[CVResult]) -> CVResult:
        """Override to provide behaviour different from the mean.
        Confidence intervals are averaged between folds.
        Since confidence intervals are averaged, and CI and std-dev are in linear relation under assumption of
        normality, we also average the standard deviations."""
        fitnesses = []
        std_devs = []
        cis = []
        importances_list = []
        bootstrap_means = []
        for r in fold_results:
            fitnesses.append(r.fitness())
            if r.has_std_dev():
                std_devs.append(r.std_dev())
            if r.has_ci95():
                cis.append(r.ci95())
            if r.has_importances():
                importances_list.append(r.importances())
            if r.has_bootstrap_mean():
                bootstrap_means.append(r.bootstrap_mean())
        if len(importances_list) == 0:
            importances = None
        else:
            importances = vector_mean(vectors=importances_list)
        n_folds = len(fold_results)
        if len(std_devs) == n_folds:
            std_dev = std_dev_of_uncorrelated_mean(std_devs=std_devs)
        else:
            std_dev = None
        if len(cis) == n_folds:
            ci = confidence_interval_of_uncorrelated_mean(confidence_intervals=cis)
        else:
            ci = None
        if len(bootstrap_means) == n_folds:
            boot_mean = KahanSummer.mean(elems=bootstrap_means)
        else:
            boot_mean = None
        return CVResult(fitness=KahanSummer.mean(elems=fitnesses),
                        std_dev=std_dev,
                        ci95=ci,
                        importances=importances,
                        bootstrap_mean=boot_mean)

    @abstractmethod
    def compute_from_structure(self, hyperparams, hp_manager: Optional[MvHyperparamManager],
                               data: Optional[InputData]) -> CVResult:
        """x already filtered by columns if necessary."""
        raise NotImplementedError()


class ClassificationObjectiveComputer(ObjectiveComputer, ABC):

    def is_survival_objective_computer(self) -> bool:
        return False

    def is_structural_objective_computer(self) -> bool:
        return False

    def can_compute_from_classes(self) -> bool:
        return True

    def compute_from_predictor_and_test_mv(
            self, predictor: MVPredictor, test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None) -> CVResult:
        y_pred = predictor.predict(test_data.views())
        if len(y_pred) != test_data.n_samples():
            raise ValueError("Different number of labels and predictions.\n" +
                             "Labels: " + str(test_data.n_samples()) + "\n" +
                             "Predictions: " + str(len(y_pred)) + "\n")
        return self.compute_from_classes_mv(
            train_pred=None, train_true=None,
            test_pred=y_pred, test_true=test_data.outcome_data(),
            hyperparams=None, hp_manager=None)

    @staticmethod
    def requires_training_predictions():
        """
        If true requires that training predictions and actual values are passed. Otherwise, it works also without them.
        An objective that works without training data can be computed on training data using it in place of the
        testing data."""
        return False

    @staticmethod
    def requires_target() -> bool:
        return True

    @staticmethod
    def requires_predictions():
        return True

    def compute_from_structure(self, hyperparams, hp_manager: Optional[MvHyperparamManager],
                               data: Optional[ModelReadyInputData]) -> CVResult:
        raise IllegalStateError()


class CrispClassificationObjectiveComputer(ClassificationObjectiveComputer, ABC):

    def is_crisp_class_objective_computer(self) -> bool:
        return True

    def is_proba_class_objective_computer(self) -> bool:
        return False


class ProbaClassificationObjectiveComputer(ClassificationObjectiveComputer, ABC):

    def is_crisp_class_objective_computer(self) -> bool:
        return False

    def is_proba_class_objective_computer(self) -> bool:
        return True
