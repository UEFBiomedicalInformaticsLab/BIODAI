import math
from typing import Sequence, Optional

import sklearn
from numpy import ravel, ones, number, array
from pandas import DataFrame

from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from univariate_feature_selection.univariate_feature_selection import filter_any_nan_mask, filter_low_variance_mask, \
    LOW_VARIANCE
from univariate_property_computer.parallel_univariate_property_computer import compute_univariate_property_with_workers
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.list_math import list_and
from util.math.online_variance_builder import OnlineVarianceBuilder
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table
from util.utils import p_adjust_bh


class AnovaPValComputer(UnivariatePropertyComputer):

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,

    def inner_compute_property(
            self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None) -> float:
        b = OnlineVarianceBuilder()
        for f in feature:
            if math.isnan(f):
                return math.nan
            b.add(float(f))
        if b.unbiased_variance() <= LOW_VARIANCE:
            return math.nan
            # Anova does not work with nan or almost zero variance, so we filter out these features,
            # that will get 0 probability in the resulting distribution.
        x = array(feature).reshape(-1, 1)
        anova_res = sklearn.feature_selection.f_classif(X=x, y=outcome.first_col())
        p_val = anova_res[1][0]
        return p_val

    def nick(self) -> str:
        return "anova"



class FeatureImportanceAnova(FeatureImportance):

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        p_vals = compute_univariate_property_with_workers(
            single_feature_computer=AnovaPValComputer(), data=x, outcome=y,
            n_proc=n_proc, task_name="ANOVA", printer=printer)
        mask = [not math.isnan(p) for p in p_vals]
        valid_p_vals = []
        for i in range(len(mask)):
            if mask[i]:
                valid_p_vals.append(p_vals[i])
        fdr = p_adjust_bh(valid_p_vals)
        res = ones(len(mask))
        res[mask] = fdr
        res = 1.0 - res
        return ConcreteDistribution(probs=res)

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        y = ravel(y)
        # Anova does not work with nan or almost zero variance, so we filter out these features,
        # that will get 0 probability in the resulting distribution.
        mask = list_and(filter_any_nan_mask(x), filter_low_variance_mask(x))
        p_values = []
        for i in range(len(mask)):
            if mask[i]:
                anova_res = sklearn.feature_selection.f_classif(X=x[x.columns[[i]]], y=y)
                p_vals = anova_res[1]
                p_values.append(p_vals[0])
        fdr = p_adjust_bh(p_values)
        res = ones(len(mask))
        res[mask] = fdr
        res = 1.0 - res
        return ConcreteDistribution(probs=res)

    def nick(self) -> str:
        return "anovaFI"

    def name(self) -> str:
        return "anova FI"

    def __str__(self) -> str:
        return "anova feature importance"
