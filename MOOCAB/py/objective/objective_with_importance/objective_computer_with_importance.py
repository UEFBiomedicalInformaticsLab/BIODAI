from abc import ABC, abstractmethod
from collections.abc import Sequence
from statistics import fmean
from typing import Union

import numpy as np
from numpy import ravel
from pandas import DataFrame
from sklearn.metrics import confusion_matrix

from bootstrap.bootstrap_distribution import bootstrap_ci95, bootstrap_ci95_from_classes,\
    bootstrap_ci95_from_structure, bootstrap_ci95_all
from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Model, Predictor
from multi_view_utils import filter_by_mask
from objective.objective_computer import ObjectiveComputer, ClassificationObjectiveComputer
from objective.objective_with_importance.feature_importance_by_permutation import feature_importance_by_permutation
from util.hyperbox.hyperbox import ConcreteInterval
from util.math.list_math import list_add_all
from util.math.mean_builder import KahanMeanBuilder
from util.math.utils import std_dev_of_uncorrelated_mean, confidence_interval_of_uncorrelated_mean
from util.randoms import random_seed
from util.math.summer import KahanSummer
from util.uniform_list import UniformList
from util.utils import IllegalStateError


DEFAULT_N_RESAMPLES = 200
DEFAULT_STRUCTURAL_N_RESAMPLES = max(int(DEFAULT_N_RESAMPLES/20.0), 1)  # Less resamples because more expensive.
DEFAULT_FROM_PREDICTORS_N_RESAMPLES = DEFAULT_STRUCTURAL_N_RESAMPLES  # Less resamples because more expensive.


class ObjectiveComputerWithImportance(ObjectiveComputer, ABC):

    @abstractmethod
    def _compute_with_kfold_cv_class_with_importance(
            self, model: Model, x, y: DataFrame, folds_list,
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x includes only the features to actually use."""
        raise NotImplementedError()

    def compute_with_kfold_cv_with_importance(
            self, model: Model, x, y: DataFrame, folds_list,
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x and y need to be already selected. x already filtered by columns if necessary.
        Returned distribution defaults to uniform distribution if computing feature importance is not supported.
        It is assumed that the test sets are a partition of all the samples seen by this procedure."""
        if not self.requires_predictions():
            raise IllegalStateError("This method is for objectives with models and predictions.")
        if self.is_class_objective_computer() and not self.force_general_cv():
            return self._compute_with_kfold_cv_class_with_importance(
                model=model, x=x, y=y, folds_list=folds_list, compute_fi=compute_fi,
                compute_confidence=compute_confidence)
        else:
            return self._compute_with_kfold_cv_general_with_importance(
                model=model, x=x, y=y, folds_list=folds_list, compute_fi=compute_fi,
                compute_confidence=compute_confidence)

    def _compute_with_kfold_cv_general_with_importance(
            self, model: Model, x, y, folds_list,
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x includes only the features to actually use."""
        results = []
        for fold in folds_list:
            x_train, y_train, x_test, y_test = self._fold_data(x, y, fold)
            fit_model = model.fit(x=x_train, y=y_train)
            result = self.compute_from_predictor_and_test(
                predictor=fit_model, x_test=x_test, y_test=y_test)
            if compute_fi:
                result.set_importances(
                    importances=self.feature_importance(predictor=fit_model, x_test=x_test, y_test=y_test))
            if compute_confidence:
                ci, std_dev, b_mean = bootstrap_ci95(
                    objective_computer=self, predictor=fit_model, x_test=x_test, y_test=y_test,
                    n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
                result.set_ci95(ci95=ci)
                result.set_std_dev(std_dev=std_dev)
                result.set_bootstrap_mean(bootstrap_mean=b_mean)
            results.append(result)
        return self._combine_fold_results(fold_results=results)

    def feature_importance(self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame) -> list[float]:
        return feature_importance_by_permutation(
            objective_computer=self, predictor=predictor, x_test=x_test, y_test=y_test, seed=random_seed())

    def compute_from_structure_with_importance(self, hyperparams, hp_manager: Union[HyperparamManager, None],
                                               x: DataFrame, y: DataFrame,
                                               compute_fi: bool = False,
                                               compute_confidence: bool = False) -> CVResult:
        """x already filtered by columns if necessary."""
        main_res = self.compute_from_structure(hyperparams=hyperparams, hp_manager=hp_manager,
                                               x=x, y=y)
        if compute_fi:
            main_res.set_importances(UniformList(value=0.0, size=hp_manager.n_active_features(hyperparams=hyperparams)))
        if compute_confidence:
            if self.requires_target():
                ci, sd, b_mean = bootstrap_ci95_from_structure(
                    objective_computer=self,
                    hyperparams=hyperparams, hp_manager=hp_manager,
                    x_test=x, y_test=y, n_resamples=DEFAULT_STRUCTURAL_N_RESAMPLES)
                main_res.set_std_dev(sd)
                main_res.set_ci95(ci)
                main_res.set_bootstrap_mean(bootstrap_mean=b_mean)
            else:  # Does not require target.
                fitness = main_res.fitness()
                main_res.set_std_dev(0.0)
                main_res.set_ci95(ConcreteInterval(fitness, fitness))
                main_res.set_bootstrap_mean(bootstrap_mean=0.0)
        return main_res

    def compute_from_structure_with_importance_all(
            self, hyperparams_seq: Sequence, hp_manager: Union[HyperparamManager, None],
            x: DataFrame, y: DataFrame,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> Sequence[CVResult]:
        return [self.compute_from_structure_with_importance(
            hyperparams=h,
            hp_manager=hp_manager,
            x=filter_by_mask(x=x, mask=h),
            y=y,
            compute_fi=compute_fi, compute_confidence=compute_confidence) for h in hyperparams_seq]

    def compute_from_predictor_and_test_with_importance(
            self, predictor: Predictor, x_test: DataFrame, y_test: DataFrame,
            x_train: Union[DataFrame, None] = None,
            y_train: Union[DataFrame, None] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        result = self.compute_from_predictor_and_test(
            predictor=predictor, x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train)
        if compute_fi:
            result.set_importances(
                importances=self.feature_importance(predictor=predictor, x_test=x_test, y_test=y_test))
        if compute_confidence:
            ci, std_dev, b_mean = bootstrap_ci95(
                objective_computer=self, predictor=predictor, x_test=x_test, y_test=y_test,
                n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
            result.set_ci95(ci95=ci)
            result.set_std_dev(std_dev=std_dev)
            result.set_bootstrap_mean(bootstrap_mean=b_mean)
        return result

    def compute_from_predictor_and_test_with_importance_all(
            self, predictors: Sequence[Predictor], x_test: DataFrame, y_test: DataFrame,
            x_train: Union[DataFrame, None] = None,
            y_train: Union[DataFrame, None] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> Sequence[CVResult]:
        """Faster than calling iteratively for each predictor because the bootstrapped pools are created
        only once for all the predictors."""
        results = [self.compute_from_predictor_and_test(
            predictor=p, x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train) for p in predictors]
        if compute_fi:
            for i in range(len(results)):
                results[i].set_importances(
                    importances=self.feature_importance(predictor=predictors[i], x_test=x_test, y_test=y_test))
        if compute_confidence:
            bootstrap_res = bootstrap_ci95_all(
                objective_computer=self, predictors=predictors, x_test=x_test, y_test=y_test,
                n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
            for i in range(len(results)):
                res = results[i]
                boot_res = bootstrap_res[i]
                res.set_ci95(ci95=boot_res[0])
                res.set_std_dev(std_dev=boot_res[1])
                res.set_bootstrap_mean(bootstrap_mean=boot_res[2])
        return results


class ClassificationObjectiveComputerWithImportance(
        ObjectiveComputerWithImportance, ClassificationObjectiveComputer, ABC):

    def _compute_with_kfold_cv_class_with_importance(
            self, model: Model, x, y: DataFrame, folds_list,
            compute_fi: bool = True, compute_confidence: bool = False,
            bootstrap_on_whole: bool = False) -> CVResult:
        """Importances are still computed on a fold by fold basis and then averaged, assuming that predictions
        on training do not matter. If predictions on training are needed the method will fail with an exception.
        If bootstrap on whole is true, the resampling happens on the union of the samples from all folds.
        Otherwise, resampling is applied to each fold and the confidence intervals are combined assuming normality."""
        pred_y_train = []
        true_y_train = []
        pred_y_test = []
        true_y_test = []
        imps = []
        fold_cis = []
        fold_std_devs = []
        fold_fitnesses = []
        fold_bootstrap_means = []
        for fold in folds_list:
            x_train, y_train, x_test, y_test = self._fold_data(x, y, fold)
            fit_model = model.fit(x=x_train, y=y_train)
            fold_pred_y_train = fit_model.predict(x_train)
            pred_y_train.extend(fold_pred_y_train)
            fold_true_y_train = ravel(y_train)
            true_y_train.extend(fold_true_y_train)
            fold_pred_y_test = fit_model.predict(x_test)
            pred_y_test.extend(fold_pred_y_test)
            fold_true_y_test = ravel(y_test)
            true_y_test.extend(fold_true_y_test)
            if compute_fi:
                imps.append(self.feature_importance(predictor=fit_model, x_test=x_test, y_test=y_test))
            if compute_confidence and not bootstrap_on_whole:
                ci, sd, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=self, pred_y_test=fold_pred_y_test, true_y_test=fold_true_y_test,
                    n_resamples=DEFAULT_N_RESAMPLES, pred_y_train=fold_pred_y_train, true_y_train=fold_true_y_train)
                fold_cis.append(ci)
                fold_std_devs.append(sd)
                fold_bootstrap_means.append(b_mean)
                fold_fitnesses.append(self.compute_from_classes(
                    hyperparams=None, hp_manager=None,
                    test_pred=fold_pred_y_test, test_true=fold_true_y_test,
                    train_pred=fold_pred_y_train, train_true=fold_true_y_train).fitness())
        cv_result = self.compute_from_classes(
            hyperparams=None, hp_manager=None,
            test_pred=pred_y_test, test_true=true_y_test,
            train_pred=pred_y_train, train_true=true_y_train)
        if compute_fi:
            cv_result.set_importances(importances=list_add_all(lists=imps))
        if compute_confidence:
            if bootstrap_on_whole:  # We compute confidence on the merged predictions.
                ci, std_dev, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=self, pred_y_test=pred_y_test, true_y_test=true_y_test,
                    n_resamples=DEFAULT_N_RESAMPLES, pred_y_train=pred_y_train, true_y_train=true_y_train)
                cv_result.set_ci95(ci95=ci)
                cv_result.set_std_dev(std_dev=std_dev)
                cv_result.set_bootstrap_mean(bootstrap_mean=b_mean)
            else:
                cv_result.set_std_dev(std_dev_of_uncorrelated_mean(std_devs=fold_std_devs))
                cv_result.set_ci95(confidence_interval_of_uncorrelated_mean(confidence_intervals=fold_cis))
                cv_result.set_bootstrap_mean(bootstrap_mean=KahanSummer.mean(fold_bootstrap_means))
        return cv_result

    def compute_from_classes_with_importance(
            self, hyperparams, hp_manager: Union[HyperparamManager, None],
            test_pred, test_true,
            train_pred=None, train_true=None,
            compute_confidence: bool = False) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data.
        Throws exception if not applicable for this objective."""
        raise NotImplementedError()

    def compute_from_structure_with_importance(
            self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
            y: DataFrame, compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()

    def is_structural_objective_computer(self) -> bool:
        return False


class Accuracy(ClassificationObjectiveComputerWithImportance):

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        accuracy = sum(diag) / len_y
        return CVResult(fitness=accuracy)

    def nick(self):
        return "accuracy"

    @staticmethod
    def requires_predictions():
        return True


class BalancedAccuracy(ClassificationObjectiveComputerWithImportance):

    def __compute_from_classes_old(
            self, hyperparams, hp_manager: HyperparamManager,
            test_pred, test_true, train_pred=None, train_true=None) -> CVResult:
        """This method is slower than the new one that does not create the full confusion matrix.
        If a class has 0 elements it is ignored and the mean is computed on the other classes."""
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        # FP, FN, TP and TN are normalized so they sum to 1.
        fn = (confusion_m.sum(axis=1) - diag) / len_y
        tp = diag / len_y
        recall_den = tp + fn
        recalls = []
        for i in range(tp.size):
            den = recall_den[i]
            if den > 0:
                recalls.append(tp[i]/den)
        return CVResult(fitness=KahanSummer.mean(recalls))

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred: Sequence, test_true: Sequence, train_pred=None, train_true=None) -> CVResult:
        """If a class has 0 elements it is ignored and the mean is computed on the other classes."""
        if isinstance(test_pred, DataFrame):  # If dataframe get the first column
            test_pred = test_pred.iloc[:, 0]
        if isinstance(test_true, DataFrame):
            test_true = test_true.iloc[:, 0]
        # test_pred = ravel(test_pred)   # ravel works but is slow.
        # test_true = ravel(test_true)
        n_samples = len(test_pred)
        if len(test_true) != n_samples:
            raise ValueError()
        tp = dict()
        fn = dict()
        pred_i = iter(test_pred)
        for truth in test_true:
            if next(pred_i) == truth:
                tp[truth] = tp.get(truth, 0) + 1
            else:
                fn[truth] = fn.get(truth, 0) + 1
        labels = set().union(*[tp, fn])
        mean_builder = KahanMeanBuilder()
        for label in labels:
            label_tp = tp.get(label, 0)
            mean_builder.add(label_tp/(label_tp+fn.get(label, 0)))
        return CVResult(fitness=mean_builder.mean())

    def nick(self) -> str:
        return "bal_acc"

    def name(self) -> str:
        return "balanced accuracy"

    @staticmethod
    def requires_predictions():
        return True


class MacroF1(ClassificationObjectiveComputerWithImportance):

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
                             test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
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
        return CVResult(fitness=fmean(fscores))

    def nick(self) -> str:
        return "macro-F1"

    def __str__(self):
        return "macro-averaged F1-score"

    @staticmethod
    def requires_predictions():
        return True
