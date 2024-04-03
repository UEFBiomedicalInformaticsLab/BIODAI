from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Union

from numpy import ravel
from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor, Model
from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.math.list_math import vector_mean
from util.math.utils import std_dev_of_uncorrelated_mean, confidence_interval_of_uncorrelated_mean
from util.named import NickNamed
from util.math.summer import KahanSummer
from util.utils import IllegalStateError


class ObjectiveComputer(NickNamed, ABC):

    @abstractmethod
    def compute_from_predictor_and_test(self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        """x includes only the features to actually use.
        This method might fail if also training data is needed but not passed,
        or if the hyperparameters are needed."""
        raise NotImplementedError()

    @abstractmethod
    def is_class_objective_computer(self) -> bool:
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
    def compute_from_classes(
            self, hyperparams, hp_manager: Union[HyperparamManager, None],
            test_pred, test_true,
            train_pred=None, train_true=None) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data.
        Sets also feature importances as uniformly 0.
        Throws exception if not applicable for this objective."""
        raise NotImplementedError()

    @abstractmethod
    def _compute_with_kfold_cv_class(
            self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        raise NotImplementedError()

    def compute_with_kfold_cv(
            self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        """x and y need to be already selected. x already filtered by columns if necessary.
        Returned distribution defaults to uniform distribution if computing feature importance is not supported.
        It is assumed that the test sets are a partition of all the samples seen by this procedure."""
        if not self.requires_predictions():
            raise IllegalStateError("This method is for objectives with models and predictions.")
        if self.is_class_objective_computer() and not self.force_general_cv():
            return self._compute_with_kfold_cv_class(
                model=model, x=x, y=y, folds_list=folds_list)
        else:
            return self._compute_with_kfold_cv_general(
                model=model, x=x, y=y, folds_list=folds_list)

    def force_general_cv(self) -> bool:
        """Return true to force the use of general cv when classification cv would be used otherwise."""
        return False

    @staticmethod
    def _fold_data(all_x, all_y, fold):
        train_mask = fold[0]
        test_mask = fold[1]
        x_train = all_x.iloc[train_mask]
        y_train = all_y.iloc[train_mask]  # TODO This could be a generic Sequence
        x_test = all_x.iloc[test_mask]
        y_test = all_y.iloc[test_mask]
        return x_train, y_train, x_test, y_test

    def _compute_with_kfold_cv_general(
            self, model: Model, x, y, folds_list) -> CVResult:
        """x and y need to be already selected."""
        results = []
        for fold in folds_list:
            x_train, y_train, x_test, y_test = self._fold_data(x, y, fold)
            fit_model = model.fit(x=x_train, y=y_train)
            results.append(self.compute_from_predictor_and_test(
                predictor=fit_model, x_test=x_test, y_test=y_test))
        return self._combine_fold_results(fold_results=results)

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
    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None],
                               x: DataFrame, y: DataFrame) -> CVResult:
        """x already filtered by columns if necessary."""
        raise NotImplementedError()


class ClassificationObjectiveComputer(ObjectiveComputer, ABC):

    def is_class_objective_computer(self) -> bool:
        return True

    def is_survival_objective_computer(self) -> bool:
        return False

    def is_structural_objective_computer(self) -> bool:
        return False

    def can_compute_from_classes(self) -> bool:
        return True

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
                                        x_train: Union[DataFrame, None] = None,
                                        y_train: Union[DataFrame, None] = None) -> CVResult:
        y_pred = predictor.predict(x_test)
        return self.compute_from_classes(
            hyperparams=None, hp_manager=None, train_pred=None, train_true=None, test_pred=y_pred, test_true=y_test)

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

    def _compute_with_kfold_cv_class(
            self, model: Model, x, y: DataFrame, folds_list) -> CVResult:
        pred_y_train = []
        true_y_train = []
        pred_y_test = []
        true_y_test = []
        for fold in folds_list:
            x_train, y_train, x_test, y_test = self._fold_data(x, y, fold)
            fit_model = model.fit(x=x_train, y=y_train)
            pred_y_train.extend(fit_model.predict(x_train))
            true_y_train.extend(ravel(y_train))
            pred_y_test.extend(fit_model.predict(x_test))
            true_y_test.extend(ravel(y_test))
        return self.compute_from_classes(
            hyperparams=None, hp_manager=None,
            test_pred=pred_y_test, test_true=true_y_test,
            train_pred=pred_y_train, train_true=true_y_train)

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        raise IllegalStateError()
