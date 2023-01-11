import math
from abc import ABC, abstractmethod
from collections import Iterable
from statistics import fmean
import numpy as np
from numpy import ravel
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor, Model
from util.named import NickNamed
from util.summer import KahanSummer
from util.utils import IllegalStateError


class ObjectiveComputer(NickNamed, ABC):

    @abstractmethod
    def compute_from_predictor_and_test(self, predictor: Predictor, x_test, y_test) -> float:
        """This method might fail if also predictions on training set are needed,
        or if the hyperparameters are needed."""
        raise NotImplementedError()

    @abstractmethod
    def is_class_objective_computer(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_survival_objective_computer(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def requires_predictions() -> bool:
        """Override to return false if predictions are not needed."""
        return True

    @staticmethod
    @abstractmethod
    def requires_training_predictions() -> bool:
        raise NotImplementedError()

    def val_to_label(self, value: float) -> float:
        """
        Value of the objective to pretty value used in plots, logs, etc. Defaults to identity function.
            """
        return value

    def vals_to_labels(self, values: Iterable[float]) -> list[float]:
        return [self.val_to_label(x) for x in values]

    @abstractmethod
    def nick(self) -> str:
        raise NotImplementedError()

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()

    @abstractmethod
    def compute_from_classes(
            self, hyperparams, hp_manager: HyperparamManager,
            test_pred, test_true, train_pred=None, train_true=None) -> float:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data.
        Throws exception if not applicable for this objective."""
        raise NotImplementedError()

    @abstractmethod
    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> float:
        raise NotImplementedError()

    def compute_with_kfold_cv(self, model: Model, x, y: DataFrame, folds_list) -> float:
        """x and y need to be already selected. x already filtered by columns if necessary."""
        if not self.requires_predictions():
            raise IllegalStateError("This method is for objectives with models and predictions.")
        if self.is_class_objective_computer() and not self.force_general_cv():
            return self._compute_with_kfold_cv_class(model=model, x=x, y=y, folds_list=folds_list)
        else:
            return self._compute_with_kfold_cv_general(model=model, x=x, y=y, folds_list=folds_list)

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

    def _compute_with_kfold_cv_general(self, model: Model, x, y, folds_list) -> float:
        results = []
        for fold in folds_list:
            x_train, y_train, x_test, y_test = self._fold_data(x, y, fold)
            fit_model = model.fit(x=x_train, y=y_train)
            results.append(self.compute_from_predictor_and_test(predictor=fit_model, x_test=x_test, y_test=y_test))
        return self._combine_fold_results(fold_results=results)

    def _combine_fold_results(self, fold_results: [float]) -> float:
        """Override to provide behaviour different from the mean"""
        return KahanSummer.mean(elems=fold_results)


class ClassificationObjectiveComputer(ObjectiveComputer, ABC):

    def is_class_objective_computer(self) -> bool:
        return True

    def is_survival_objective_computer(self) -> bool:
        return False

    def compute_from_predictor_and_test(self, predictor: Predictor, x_test, y_test) -> float:
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

    def _compute_with_kfold_cv_class(self, model: Model, x, y: DataFrame, folds_list) -> float:
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


class Leanness(ClassificationObjectiveComputer):

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true, train_pred, train_true) -> float:
        if hp_manager is None:
            raise ValueError("hp_manager is None")
        n_features = hp_manager.n_active_features(hyperparams)
        return 1/(1+n_features)

    def nick(self):
        return "leanness"

    @staticmethod
    def requires_predictions():
        return False

    def val_to_label(self, value) -> float:
        """ Not rounded to int since passed value can be an average. """
        if value <= 0.0:
            return math.inf
        else:
            return (1.0 / value) - 1.0


class SoftLeanness(ClassificationObjectiveComputer):

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true, train_pred, train_true) -> float:
        if hp_manager is None:
            raise ValueError("hp_manager is None")
        n_features = hp_manager.n_active_features(hyperparams)
        return self.compute_from_n_features(n_features)

    def nick(self):
        return "soft_leanness"

    @staticmethod
    def requires_predictions():
        return False

    def val_to_label(self, value) -> float:
        """ Not rounded to int since passed value can be an average. """
        if value <= 0.0:
            return math.inf
        else:
            return ((1.0 / value) - 1.0)**2

    @staticmethod
    def compute_from_n_features(n_features: int) -> float:
        return 1.0/(1.0+math.sqrt(n_features))

    @staticmethod
    def leanness_to_soft_leanness(leanness: float) -> float:
        return SoftLeanness.compute_from_n_features(round(Leanness().val_to_label(leanness)))


class Accuracy(ClassificationObjectiveComputer):

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true, train_pred, train_true) -> float:
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        accuracy = sum(diag) / len_y
        return accuracy

    def nick(self):
        return "accuracy"

    @staticmethod
    def requires_predictions():
        return True


class BalancedAccuracy(ClassificationObjectiveComputer):

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true, train_pred=None, train_true=None) -> float:
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        # FP, FN, TP and TN are normalized so they sum to 1.
        fn = (confusion_m.sum(axis=1) - diag) / len_y
        tp = diag / len_y
        recall = tp / (tp + fn)
        return np.mean(recall)

    def nick(self) -> str:
        return "bal_acc"

    def name(self) -> str:
        return "balanced accuracy"

    @staticmethod
    def requires_predictions():
        return True


class MacroF1(ClassificationObjectiveComputer):

    @staticmethod
    def single_class_fscore(fp, fn, tp):
        """We replace Nan with 0 that is the lowest possible value."""
        if tp == 0:
            return 0.0  # precision + recall is 0
        den_precision = tp + fp
        if den_precision == 0:
            return 0.0
        den_recall = tp + fn
        if den_recall == 0:
            return 0.0
        precision = tp / den_precision
        recall = tp / den_recall
        return (2.0 * precision * recall) / (precision + recall)

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true, train_pred, train_true) -> float:
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        # FP, FN, TP and TN are normalized so they sum to 1.
        fp = (confusion_m.sum(axis=0) - diag) / len_y
        fn = (confusion_m.sum(axis=1) - diag) / len_y
        tp = diag / len_y
        fscores = []
        for fp_i, fn_i, tp_i in zip(fp, fn, tp):
            fscores.append(MacroF1.single_class_fscore(fp_i, fn_i, tp_i))
        return fmean(fscores)

    def nick(self) -> str:
        return "macro-F1"

    def __str__(self):
        return "macro-averaged F1-score"

    @staticmethod
    def requires_predictions():
        return True
