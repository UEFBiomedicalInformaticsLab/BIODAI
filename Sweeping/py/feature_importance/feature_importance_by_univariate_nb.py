from collections.abc import Sequence
from typing import Optional

import warnings
import numpy as np
from numpy import number
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline

from descriptor.descriptor import Descriptor
from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome, CategoricalOutcome
from input_data.outcome_type import OutcomeType
from univariate_property_computer.parallel_univariate_property_computer import compute_univariate_property_with_workers
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from util.fast_auc import binary_auc, multiclass_ovr_auc, normalize_auc_to_importance
from util.math.summer import KahanSummer
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.distribution.distribution import Distribution, ConcreteDistribution


DEFAULT_WEIGHTED = False
NB_NICK = "nb"


def safe_roc_auc_importance(
    y_true,
    y_score,
    *,
    weighted: bool = DEFAULT_WEIGHTED,
    **kwargs
) -> float:
    """
    Compute a normalized AUC importance score:
    - Returns 0.0 if predictions contain NaNs or if AUC is not computable.
    - Binary: 1D score vector, or (n,2) matrix -> uses column 1 as positive class.
    - Multiclass: (n,C) matrix -> OvR with 'weighted' or 'macro' averaging.
    - Normalization: max(0, 2*AUC - 1) to [0,1].
    kwargs adds robustness in case sklearn wants to pass other arguments.
    """
    # Normalize to arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Early bail on NaNs if numeric
    if y_score.dtype.kind in "fc" and np.isnan(y_score).any():
        return 0.0

    try:
        if y_score.ndim == 1:
            # Binary 1D scores: assumes scores are for the "positive" class
            auc = binary_auc(y_true, y_score)
        elif y_score.ndim == 2:
            if y_score.shape[1] == 2:
                # Binary predict_proba-like: take positive column (index 1)
                auc = binary_auc(y_true, y_score[:, 1])
            else:
                # Multiclass OvR: shape (n_samples, n_classes)
                average = "weighted" if weighted else "macro"
                auc = multiclass_ovr_auc(y_true, y_score, average=average, classes=None)
        else:
            # e.g., only one class present in y_true in a CV split
            return 0.0
    except ValueError:
        return 0.0

    # Normalize AUC to [0, 1]
    return normalize_auc_to_importance(auc)



class NBComputer(UnivariatePropertyComputer):
    __weighted: bool
    __clf: Pipeline
    __cv_splitter: KFold

    def __init__(self, weighted: bool = DEFAULT_WEIGHTED):
        UnivariatePropertyComputer.__init__(self=self, descriptor=NB_COMPUTER_DESCRIPTOR)
        self.__weighted = weighted
        # Create a pipeline: imputer + GaussianNB
        self.__clf = make_pipeline(SimpleImputer(strategy='mean'), GaussianNB())
        self.__cv_splitter = KFold(n_splits=2, shuffle=False)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,

    def inner_compute_property(
            self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None) -> float:
        X = np.asarray(feature).reshape(-1, 1)  # Should be faster than using a dataframe
        y = np.asarray(outcome.first_col())  # Must be an array to use NumPy indexing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = []
            for train_idx, test_idx in self.__cv_splitter.split(X):
                # Instead of cloning, just re-fit the same estimator, this avoids expensive checks by SKLearn.
                self.__clf.fit(X[train_idx], y[train_idx])
                proba = self.__clf.predict_proba(X[test_idx])
                scores.append(safe_roc_auc_importance(y[test_idx], proba, weighted=self.__weighted))
        return KahanSummer.mean(scores)

    def nick(self) -> str:
        return NB_NICK


class NBComputerDescriptor(Descriptor):

    def nick(self) -> str:
        return NB_NICK


NB_COMPUTER_DESCRIPTOR = NBComputerDescriptor()


class FeatureImportanceUnivariateNB(FeatureImportance):
    __weighted: bool
    __verbose: bool

    def __init__(self, weighted: bool = DEFAULT_WEIGHTED, verbose: bool = False):
        self.__weighted = weighted
        self.__verbose = verbose

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        scores = compute_univariate_property_with_workers(
            single_feature_computer=NBComputer(weighted=self.__weighted), data=x, outcome=y,
            n_proc=n_proc, task_name="NB AUC", printer=printer)
        if self.__verbose:
            print("Num scores: " + str(len(scores)))
            print("Scores sum: " + str(sum(scores)))
            print("Nonzero scores: " + str(sum([s > 0.0 for s in scores])))
        return ConcreteDistribution(probs=scores)

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        return self.compute(
            x=BackedTable(backend=NpTable(data=x)), y=CategoricalOutcome(data=y, name="Unnamed outcome"),
            printer=printer)

    def nick(self) -> str:
        return "uniNBFI"

    def name(self) -> str:
        return "univariate NB AUC FI"

    def __str__(self) -> str:
        return "univariate NB AUC feature importance"
