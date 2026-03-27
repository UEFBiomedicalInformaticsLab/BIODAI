from abc import ABC, abstractmethod
from collections.abc import Sequence
from statistics import fmean
from typing import Optional, Any

import numpy as np
from numpy import ravel
from pandas import DataFrame
from sklearn.metrics import confusion_matrix

from bootstrap.bootstrap_distribution import bootstrap_ci95_from_classes, \
    bootstrap_ci95_from_structure, bootstrap_ci95_all_mv, bootstrap_ci95_mv, \
    DEFAULT_RESAMPLING_SEED
from cross_validation.single_objective.cv_result import CVResult
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from model.sv_model import PredictProbaResult
from objective.objective_computer import ObjectiveComputer, CrispClassificationObjectiveComputer, \
    ClassificationObjectiveComputer, ProbaClassificationObjectiveComputer
from objective.objective_with_importance.feature_importance_by_permutation import feature_importance_by_permutation_mv
from util.fast_auc import binary_auc, multiclass_ovr_auc
from util.hyperbox.hyperbox import ConcreteInterval
from util.math.list_math import list_add_all
from util.math.mean_builder import KahanMeanBuilder
from util.math.utils import std_dev_of_uncorrelated_mean, confidence_interval_of_uncorrelated_mean
from util.randoms import random_seed
from util.math.summer import KahanSummer
from util.uniform_list import UniformList
from util.utils import IllegalStateError
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


DEFAULT_N_RESAMPLES = 200
DEFAULT_STRUCTURAL_N_RESAMPLES = max(int(DEFAULT_N_RESAMPLES/20.0), 1)  # Less resamples because more expensive.
DEFAULT_FROM_PREDICTORS_N_RESAMPLES = DEFAULT_STRUCTURAL_N_RESAMPLES  # Less resamples because more expensive.

BRIER_NICK = "brier"
BALANCED_ACCURACY_NICK = "bal_acc"
AUC_NICK = "AUC"
AUCPR_NICK = "AUCPR"


class ObjectiveComputerWithImportance(ObjectiveComputer, ABC):

    @abstractmethod
    def _compute_with_kfold_cv_class_with_importance_mv(
            self, model: MVModel, data: ModelReadyInputData, folds_list: list[tuple[IndexArray, IndexArray]],
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x includes only the features to actually use."""
        raise NotImplementedError()

    def compute_with_kfold_cv_with_importance_mv(
            self, model: MVModel, data: ModelReadyInputData, folds_list: list[tuple[IndexArray, IndexArray]],
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x and y need to be already selected. x already filtered by columns if necessary.
        Returned distribution defaults to uniform distribution if computing feature importance is not supported.
        It is assumed that the test sets are a partition of all the samples seen by this procedure."""
        if not self.requires_predictions():
            raise IllegalStateError("This method is for objectives with models and predictions.")
        if self.is_crisp_class_objective_computer() and not self.force_general_cv():
            # Maybe this will work also with proba computers in the future.
            return self._compute_with_kfold_cv_class_with_importance_mv(
                model=model, data=data, folds_list=folds_list, compute_fi=compute_fi,
                compute_confidence=compute_confidence)
        else:
            return self._compute_with_kfold_cv_general_with_importance_mv(
                model=model, data=data, folds_list=folds_list, compute_fi=compute_fi,
                compute_confidence=compute_confidence)

    def _compute_with_kfold_cv_general_with_importance_mv(
            self, model: MVModel, data: ModelReadyInputData, folds_list: list[tuple[IndexArray, IndexArray]],
            compute_fi: bool = False, compute_confidence: bool = False) -> CVResult:
        """x includes only the features to actually use."""
        results = []
        for fold in folds_list:
            train_data = data.select_samples(row_indices=fold[0])
            test_data = data.select_samples(row_indices=fold[1])
            fit_model = model.fit(data=train_data)
            result = self.compute_from_predictor_and_test_mv(
                predictor=fit_model, test_data=test_data, train_data=train_data)
            if compute_fi:
                result.set_importances(
                    importances=self.feature_importance_mv(predictor=fit_model, test_data=test_data))
            if compute_confidence:
                ci, std_dev, b_mean = bootstrap_ci95_mv(
                    objective_computer=self, predictor=fit_model, test_data=test_data,
                    n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
                result.set_ci95(ci95=ci)
                result.set_std_dev(std_dev=std_dev)
                result.set_bootstrap_mean(bootstrap_mean=b_mean)
            results.append(result)
        return self._combine_fold_results(fold_results=results)

    def feature_importance_mv(
            self, predictor: MVPredictor, test_data: ModelReadyInputData) -> dict[str,Sequence[float]]:
        return feature_importance_by_permutation_mv(
            objective_computer=self, predictor=predictor, test_data=test_data, seed=random_seed())

    def compute_from_structure_with_importance(
            self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
            data: Optional[EvaluationReadyInputData] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        """x already filtered by columns if necessary."""
        main_res = self.compute_from_structure(hyperparams=hyperparams, hp_manager=hp_manager, data=data)
        if compute_fi:
            main_res.set_importances(
                UniformList(value=0.0, size=hp_manager.n_predictive_features(hyperparams=hyperparams)))
        if compute_confidence:
            if self.requires_target():
                ci, sd, b_mean = bootstrap_ci95_from_structure(
                    objective_computer=self,
                    hyperparams=hyperparams, hp_manager=hp_manager,
                    test_data=data, n_resamples=DEFAULT_STRUCTURAL_N_RESAMPLES)
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
            self,
            hyperparams_seq: Sequence,
            hp_manager: MvHyperparamManager,
            data: ModelReadyInputData,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> Sequence[CVResult]:
        """Each element of hyperparams_seq is an instance of the hyperparameters."""
        res = []
        for h in hyperparams_seq:
            h_data = data.select_features(hp_manager.used_feature_masks(hyperparams=h))
            res.append(self.compute_from_structure_with_importance(
                hyperparams=h,
                hp_manager=hp_manager,
                data=h_data,
                compute_fi=compute_fi, compute_confidence=compute_confidence))
        return res

    def compute_from_predictor_and_test_with_importance(
            self, predictor: MVPredictor,
            test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        result = self.compute_from_predictor_and_test_mv(
            predictor=predictor, test_data=test_data, train_data=train_data)
        if compute_fi:
            result.set_importances(
                importances=self.feature_importance_mv(predictor=predictor, test_data=test_data))
        if compute_confidence:
            ci, std_dev, b_mean = bootstrap_ci95_mv(
                objective_computer=self, predictor=predictor, test_data=test_data,
                n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
            result.set_ci95(ci95=ci)
            result.set_std_dev(std_dev=std_dev)
            result.set_bootstrap_mean(bootstrap_mean=b_mean)
        return result

    def compute_from_predictor_and_test_with_importance_all_mv(
            self, predictors: Sequence[MVPredictor],
            test_data: ModelReadyInputData,
            train_data: Optional[ModelReadyInputData] = None,
            compute_confidence: bool = False, seed: int = DEFAULT_RESAMPLING_SEED) -> Sequence[CVResult]:
        """Might be faster than calling iteratively for each predictor because
        in some cases the bootstrapped pools are created
        only once for all the predictors."""
        results = [self.compute_from_predictor_and_test_mv(
            predictor=p, test_data=test_data, train_data=train_data) for p in predictors]
        if compute_confidence:
            bootstrap_res = bootstrap_ci95_all_mv(
                objective_computer=self, predictors=predictors, test_data=test_data,
                n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES, seed=seed)
            for i in range(len(results)):
                res = results[i]
                boot_res = bootstrap_res[i]
                res.set_ci95(ci95=boot_res[0])
                res.set_std_dev(std_dev=boot_res[1])
                res.set_bootstrap_mean(bootstrap_mean=boot_res[2])
        return results

    def compute_from_predictor_and_test_with_importance_mv(
            self, predictor: MVPredictor,
            test_data: Optional[ModelReadyInputData],
            train_data: Optional[ModelReadyInputData] = None,
            compute_confidence: bool = False) -> CVResult:
        """If this method is called iteratively the bootstrap will be randomised differently every time.
        Consider using compute_from_predictor_and_test_with_importance_all_mv."""
        result = self.compute_from_predictor_and_test_mv(
            predictor=predictor, test_data=test_data, train_data=train_data)
        if compute_confidence:
            bootstrap_res = bootstrap_ci95_mv(
                objective_computer=self, predictor=predictor, test_data=test_data,
                n_resamples=DEFAULT_FROM_PREDICTORS_N_RESAMPLES)
            result.set_ci95(ci95=bootstrap_res[0])
            result.set_std_dev(std_dev=bootstrap_res[1])
            result.set_bootstrap_mean(bootstrap_mean=bootstrap_res[2])
        return result


class ClassificationObjectiveComputerWithImportance(
        ObjectiveComputerWithImportance, ClassificationObjectiveComputer, ABC):

    def _compute_with_kfold_cv_class_with_importance_mv(self, model: MVModel, data: ModelReadyInputData,
                                                        folds_list: list[tuple[IndexArray, IndexArray]],
                                                        compute_fi: bool = False,
                                                        compute_confidence: bool = False,
                                                        bootstrap_on_whole: bool = False) -> CVResult:
        """Importances are still computed on a fold by fold basis and then averaged, assuming that predictions
        on training do not matter. If predictions on training are needed the method will fail with an exception.
        If bootstrap on whole is true, the resampling happens on the union of the samples from all folds.
        Otherwise, resampling is applied to each fold and the confidence intervals are combined assuming normality.
        TODO Does not work for predict proba computers at the moment because PredictProbaResult are not preserved
        and are concatenated in normal sequences."""
        pred_y_train = []
        true_y_train = []
        pred_y_test = []
        true_y_test = []
        imps = []
        fold_cis = []
        fold_std_devs = []
        fold_fitnesses = []
        fold_bootstrap_means = []
        seed = random_seed()
        for fold in folds_list:
            train_data = data.select_samples(row_indices=fold[0])
            test_data = data.select_samples(row_indices=fold[1])
            fit_model = model.fit(data=train_data)
            fold_pred_y_train = fit_model.predict(train_data.views())
            pred_y_train.extend(fold_pred_y_train)
            fold_true_y_train = ravel(train_data.outcome_data())
            true_y_train.extend(fold_true_y_train)
            fold_pred_y_test = fit_model.predict(test_data.views())
            pred_y_test.extend(fold_pred_y_test)
            fold_true_y_test = ravel(test_data.outcome_data())
            true_y_test.extend(fold_true_y_test)
            if compute_fi:
                imps.append(self.feature_importance_mv(predictor=fit_model, test_data=test_data))
            if compute_confidence and not bootstrap_on_whole:
                ci, sd, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=self, pred_y_test=fold_pred_y_test, true_y_test=fold_true_y_test,
                    n_resamples=DEFAULT_N_RESAMPLES, pred_y_train=fold_pred_y_train, true_y_train=fold_true_y_train,
                    random=np.random.default_rng(seed=seed))
                fold_cis.append(ci)
                fold_std_devs.append(sd)
                fold_bootstrap_means.append(b_mean)
                fold_fitnesses.append(self.compute_from_classes_mv(
                    test_pred=fold_pred_y_test, test_true=fold_true_y_test,
                    train_pred=fold_pred_y_train, train_true=fold_true_y_train,
                    hyperparams=None, hp_manager=None).fitness())
        cv_result = self.compute_from_classes_mv(
            test_pred=pred_y_test, test_true=true_y_test,
            train_pred=pred_y_train, train_true=true_y_train,
            hyperparams=None, hp_manager=None)
        if compute_fi:
            cv_result.set_importances(importances=list_add_all(lists=imps))
        if compute_confidence:
            if bootstrap_on_whole:  # We compute confidence on the merged predictions.
                ci, std_dev, b_mean = bootstrap_ci95_from_classes(
                    objective_computer=self, pred_y_test=pred_y_test, true_y_test=true_y_test,
                    n_resamples=DEFAULT_N_RESAMPLES, pred_y_train=pred_y_train, true_y_train=true_y_train,
                    random=np.random.default_rng(seed=seed))
                cv_result.set_ci95(ci95=ci)
                cv_result.set_std_dev(std_dev=std_dev)
                cv_result.set_bootstrap_mean(bootstrap_mean=b_mean)
            else:
                cv_result.set_std_dev(std_dev_of_uncorrelated_mean(std_devs=fold_std_devs))
                cv_result.set_ci95(confidence_interval_of_uncorrelated_mean(confidence_intervals=fold_cis))
                cv_result.set_bootstrap_mean(bootstrap_mean=KahanSummer.mean(fold_bootstrap_means))
        return cv_result

    def compute_from_classes_with_importance(
            self, hyperparams, hp_manager: Optional[SvHyperparamManager],
            test_pred, test_true,
            train_pred=None, train_true=None,
            compute_confidence: bool = False) -> CVResult:
        """We pass also train y since there exist peculiar metrics using also the train.
        All passed y must be lists. The nature of an y element is problem dependent and can also be censored data.
        Throws exception if not applicable for this objective."""
        raise NotImplementedError()

    def compute_from_structure_with_importance(
            self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
            data: Optional[ModelReadyInputData] = None,
            compute_fi: bool = False,
            compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()


class CrispClassificationObjectiveComputerWithImportance(
        ClassificationObjectiveComputerWithImportance, CrispClassificationObjectiveComputer, ABC):
    pass


class ProbaClassificationObjectiveComputerWithImportance(
        ClassificationObjectiveComputerWithImportance, ProbaClassificationObjectiveComputer, ABC):
    pass


class Accuracy(CrispClassificationObjectiveComputerWithImportance):

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        confusion_m = confusion_matrix(y_true=test_true, y_pred=test_pred)
        len_y = len(test_true)
        diag = np.diag(confusion_m)
        accuracy = sum(diag) / len_y
        return CVResult(fitness=accuracy)

    def nick(self):
        return "accuracy"


class BalancedAccuracy(CrispClassificationObjectiveComputerWithImportance):

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """If a class has 0 elements it is ignored and the mean is computed on the other classes."""
        if isinstance(test_pred, DataFrame):  # If dataframe get the first column
            test_pred = test_pred.iloc[:, 0]
        if isinstance(test_true, DataFrame):
            test_true = test_true.iloc[:, 0]
        # test_pred = ravel(test_pred)   # ravel works but is slow.
        # test_true = ravel(test_true)
        n_samples = len(test_pred)
        if len(test_true) != n_samples:
            raise ValueError("Different number of labels and predictions.\n" +
                             "Labels: " + str(len(test_true)) + "\n" +
                             "Predictions: " + str(n_samples) + "\n")
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
        return BALANCED_ACCURACY_NICK

    def name(self) -> str:
        return "balanced accuracy"


class MacroF1(CrispClassificationObjectiveComputerWithImportance):

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

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
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


def proper_y(y: Sequence) -> Sequence:
    if isinstance(y, DataFrame):  # If dataframe get the first column
        y = y.iloc[:, 0]
    return y


def y_consistency_check(y_pred: Sequence, y_true: Sequence):
    n_samples = len(y_pred)
    if len(y_true) != n_samples:
        raise ValueError("Different number of labels and predictions.\n" +
                         "Labels: " + str(len(y_true)) + "\n" +
                         "Predictions: " + str(y_pred) + "\n")


class BrierScore(ProbaClassificationObjectiveComputerWithImportance):

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """If a class has 0 elements it is ignored and the mean is computed on the other classes."""
        test_pred = proper_y(y=test_pred)
        test_true = proper_y(y=test_true)
        y_consistency_check(y_pred=test_pred, y_true=test_true)
        if isinstance(test_pred, PredictProbaResult):
            classes = test_pred.classes()
            summer = KahanSummer()
            add_unchecked = summer.add_unchecked  # Local variable for faster access
            try:  # The try is outside the loop for speed.
                # The following loop can be crucial for performance.
                for p, t in zip(test_pred, test_true):
                    for c, cp in zip(classes, p):
                        if c == t:
                            add_unchecked((1 - cp) ** 2)
                            # This point is critical for performance and benefits from the unchecked version.
                        else:
                            add_unchecked(cp ** 2)
                brier = summer.get_sum() / (2 * len(test_true))
            except TypeError as e:
                raise TypeError(str(e) + "\n"
                                + "Classes: " + str(classes) + "\n"
                                + "test_pred: " + str(test_pred) + "\n"
                                + "test_true: " + str(test_true) + "\n")
            return CVResult(fitness=(1 - brier))
        else:
            raise ValueError("Expecting a test_pred object of type PredictProbaResult, but the type is different.\n" +
                             "test_pred:\n" + str(test_pred))

    def nick(self) -> str:
        return BRIER_NICK

    def name(self) -> str:
        return "1 - Brier score"


class AUC(ProbaClassificationObjectiveComputerWithImportance):
    """Computes multiclass weighted AUC using OvR strategy."""

    def compute_from_classes_mv(
            self,
            test_pred,
            test_true,
            train_pred=None,
            train_true=None,
            hyperparams: Optional[Any] = None,
            hp_manager: Optional[MvHyperparamManager] = None
    ) -> CVResult:
        """Computes multiclass weighted AUC using OvR strategy (fast, no sklearn)."""

        # Preprocess and validate inputs (keeping your current pipeline)
        test_pred = proper_y(test_pred)
        test_true = proper_y(test_true)
        y_consistency_check(test_pred, test_true)

        if not isinstance(test_pred, PredictProbaResult):
            raise TypeError(f"Expected PredictProbaResult, got {type(test_pred)}")

        classes = test_pred.classes()
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("AUC computation requires at least two classes.")

        # Convert once
        P = np.asarray(test_pred)
        y = np.asarray(test_true)

        try:
            if n_classes == 2:
                # Binary: follow your existing convention -> positive column index 1
                y_scores = P[:, 1]
                auc = binary_auc(y, y_scores)
            else:
                # Multiclass: OvR weighted, columns correspond to `classes`
                auc = multiclass_ovr_auc(y, P, average="weighted", classes=np.asarray(classes))
        except ValueError as e:
            # Preserve your error-reporting style (rich context)
            raise ValueError(
                f"Error computing weighted multiclass AUC (OvR): {e}\n"
                f"Classes: {classes}\n"
                f"test_pred: {P}\n"
                f"test_true: {y}"
            ) from None

        return CVResult(fitness=float(auc))

    def nick(self) -> str:
        return AUC_NICK

    def name(self) -> str:
        return "weighted AUC OvR"


class AUCPR(ProbaClassificationObjectiveComputerWithImportance):
    """Computes multiclass weighted AUC-PR using OvR strategy."""

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        """Computes multiclass weighted AUC-PR using OvR strategy."""

        test_pred = proper_y(test_pred)
        test_true = proper_y(test_true)
        y_consistency_check(test_pred, test_true)

        if not isinstance(test_pred, PredictProbaResult):
            raise TypeError(f"Expected PredictProbaResult, got {type(test_pred)}")

        classes = test_pred.classes()
        n_classes = len(classes)

        if n_classes < 2:
            raise ValueError("AUC-PR computation requires at least two classes.")
        try:
            if n_classes == 2:
                # Binary classification: use probability of positive class
                y_scores = [p[1] for p in test_pred]
                auc_pr = average_precision_score(test_true, y_scores, pos_label=classes[1])
            else:
                # Multiclass: binarize labels and compute weighted average
                y_true_bin = label_binarize(test_true, classes=classes).astype(np.float32)
                y_scores = np.array(test_pred, dtype=np.float32)
                auc_pr = average_precision_score(y_true_bin, y_scores, average='weighted')
        except ValueError as e:
            raise ValueError(
                f"Error computing weighted multiclass AUC-PR (OvR): {e}\n"
                f"Classes: {classes}\n"
                f"test_pred: {test_pred}\n"
                f"test_true: {test_true}"
            )
        return CVResult(fitness=auc_pr)

    def nick(self) -> str:
        return AUCPR_NICK

    def name(self) -> str:
        return "weighted AUC-PR OvR"