from typing import List

import numpy as np
from pandas import DataFrame

from univariate_feature_selection.parallel_anova import filter_anova_mask, filter_anova
from util.dataframes import has_non_finite, has_non_finite_error
from util.list_math import list_not, num_of_true, list_or
from util.printer.printer import Printer


def filter_all_nan(view):
    res = view.dropna(axis='columns', how='all')
    return res


def filter_any_nan_mask(view) -> List[bool]:
    return list_not(view.isna().any())


def filter_any_nan(view: DataFrame) -> DataFrame:
    support = filter_any_nan_mask(view)
    res = view.iloc[0:, support]
    return res


def filter_any_infinite_mask(view: DataFrame) -> List[bool]:
    return list_not(view.isin([np.inf, -np.inf]).any())


def filter_any_infinite(view: DataFrame) -> DataFrame:
    support = filter_any_infinite_mask(view)
    res = view.iloc[0:, support]
    return res


def filter_low_variance_mask(view: DataFrame, low_var: float = 9.0e-5) -> [bool]:
    return [v > low_var for v in view.var()]


def filter_low_variance(view: DataFrame, low_var: float = 9.0e-5) -> DataFrame:
    support = filter_low_variance_mask(view, low_var=low_var)
    res = view.iloc[0:, support]
    return res


def filter_zero_variance_mask(view: DataFrame) -> [bool]:
    return filter_low_variance_mask(view, low_var=0.0)


def filter_zero_variance(view: DataFrame) -> DataFrame:
    return filter_low_variance(view=view, low_var=0.0)


def filter_view(view, outcome: DataFrame, printer: Printer, p_val=0.05, n_proc=1):
    outcome_array = outcome.values.ravel()
    res = view.copy()
    printer.print("Features before filter: " + str(len(res.columns)))
    res = filter_anova(res, outcome_array, p_val=p_val, n_proc=n_proc)
    printer.print("Features after any NaN, zero variance and ANOVA filters: " + str(len(res.columns)))
    return res


def filter_views(views, outcome, printer: Printer, p_val=0.05, n_proc=1):
    res = {}
    for v in views:
        printer.print("filtering view " + v)
        res[v] = filter_view(view=views[v], outcome=outcome, p_val=p_val, printer=printer, n_proc=n_proc)
    return res


def filter_view_pre_cv(view, printer: Printer):
    res = view.copy()
    printer.print("Features before filter: " + str(len(res.columns)))
    res = filter_all_nan(res)
    printer.print("Features after all na filter: " + str(len(res.columns)))
    res = filter_zero_variance(res)
    printer.print("Features after zero variance filter: " + str(len(res.columns)))
    return res


def filter_views_pre_cv(views, printer: Printer):
    """Filters out features that would be filtered out in any case with any possible train set selection."""
    printer.print("Filtering views using whole dataset.")
    res = {}
    for v in views:
        printer.print("Filtering view " + v)
        res[v] = filter_view_pre_cv(views[v], printer=printer)
    return res


def compute_active_features_sv(view, y: DataFrame, printer: Printer, n_proc=1) -> List[bool]:
    outcome_array = y.values.ravel()
    printer.print("Features before filter: " + str(len(view.columns)))
    res = filter_anova_mask(view, outcome_array, n_proc=n_proc)
    printer.print("Features after any na, zero variance, and anova filters: " + str(sum(res)))
    return res


def compute_active_features_sv_multi_target(view, y: DataFrame, printer: Printer,
                                            postconditions: bool = False, n_proc=1) -> List[bool]:
    if not isinstance(y, DataFrame):
        raise ValueError("Expected y to be a DataFrame.\n" + str(y))
    res = [False] * len(view.columns)
    for c in y.columns:
        if y[c].dtype == np.float64:
            printer.print("Skipping output column " + str(c) + " containing floating point data.")
        else:
            printer.print("Computing active features for target column " + str(c))
            c_active = compute_active_features_sv(view=view, y=y[[c]], printer=printer, n_proc=n_proc)
            res = list_or(list_a=res, list_b=c_active)
            printer.print("Features active in view: " + str(sum(res)))
    if postconditions:
        if has_non_finite(view.iloc[:, res]):
            raise has_non_finite_error(view.iloc[:, res])
    return res


def compute_active_features_mv_multi_target(views, y: DataFrame, printer: Printer, n_proc=1) -> List[List[bool]]:
    res = []
    n_active = 0
    for v in views:
        printer.print("Filtering view " + v)
        view_res = compute_active_features_sv_multi_target(views[v], y, printer=printer, n_proc=n_proc)
        res.append(view_res)
        n_active += num_of_true(view_res)
    printer.print("Total number of active features: " + str(n_active))
    return res


def compute_active_features_mv(views, y: DataFrame, printer: Printer) -> List[List[bool]]:
    res = []
    n_active = 0
    for v in views:
        printer.print("Filtering view " + v)
        view_res = compute_active_features_sv(views[v], y, printer=printer)
        res.append(view_res)
        n_active += num_of_true(view_res)
    printer.print("Total number of active features: " + str(n_active))
    return res
