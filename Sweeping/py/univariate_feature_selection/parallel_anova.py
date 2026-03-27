import multiprocessing
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy import ravel, var, number
from pandas import DataFrame
from sklearn import feature_selection

from consts import DEFAULT_P_VAL
from util.dataframe.dataframes import nan_slice, select_cols_by_mask
from util.printer.printer import Printer, DEFAULT_PRINTER
from util.str_utils import pretty_duration



def anova_pval_one_feature_unchecked(
        x: np.ndarray, y, verbose: bool = True, ignore_warns: bool = True) -> float:
    """If input contains NaN or has 0 variance ANOVA will not work.
    x must be shaped in the numpy array required by the ANOVA."""
    try:
        if ignore_warns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                anova_res = feature_selection.f_classif(X=x, y=y)
        else:
            anova_res = feature_selection.f_classif(X=x, y=y)
        p_vals = anova_res[1]
        res = p_vals[0]
    except BaseException as e:
        if verbose:
            print("Exception encountered in ANOVA:")
            print(e)
            print("x:")
            print(x)
            rx = ravel(x)
            print("Variance: " + str(var(rx)))
            print("Setting p-value to 1.")
        res = 1.0
    return res


def anova_pval_one_feature_checked(
        x: Sequence[number], y, verbose: bool = True, ignore_warns: bool = True) -> float:
    """Imputes NaN values with the mean and checks for zero variance, otherwise f_classif does not work.
    Returns 1.0 if feature must be discarded.
    x can be any sequence of numbers, and will be reshaped in the numpy array required by the ANOVA."""
    x = np.array(x).reshape(-1, 1)

    # Compute mean and impute NaNs
    mean_val = np.nanmean(x)
    x = np.where(np.isnan(x), mean_val, x)

    # Check for zero variance
    if not (np.var(x) > 0.0):
        return 1.0

    return anova_pval_one_feature_unchecked(x=x, y=y, verbose=verbose, ignore_warns=ignore_warns)


def anova_filter_one_feature_checked(
        x: Sequence[number], y, p_val: float = DEFAULT_P_VAL, verbose: bool = True, ignore_warns: bool = True) -> bool:
    """Imputes NaN values with the mean and checks for zero variance, otherwise f_classif does not work.
    Returns False if feature must be discarded.
    x can be any sequence of numbers, and will be reshaped in the numpy array required by the ANOVA."""
    return anova_pval_one_feature_checked(x=x, y=y, verbose=verbose, ignore_warns=ignore_warns) < p_val


class OneColInput:

    def __init__(self, x: DataFrame, y: DataFrame, p_val: float, verbose: bool):
        """x and y are dataframes with features placed vertically."""
        self.x = x
        self.y = y
        self.p_val = p_val
        self.verbose = verbose


def anova_filter_one_col_input_checked(col_input: OneColInput) -> bool:
    return anova_filter_one_feature_checked(
        x=col_input.x, y=col_input.y, p_val=col_input.p_val,
        verbose=col_input.verbose)


def filter_anova_mask(
        view: DataFrame, outcome, p_val: float = DEFAULT_P_VAL, n_proc: int = 1, verbose: bool = False,
        printer: Printer = DEFAULT_PRINTER) -> list[bool]:
    """Checks also for NaN and zero variance otherwise f_classif does not work."""
    n_cols = len(view.columns)
    cpu_count = multiprocessing.cpu_count()
    proc_to_use = max(1, min(n_proc, cpu_count, n_cols))
    start_time = None
    if verbose:
        start_time = time.time()
        printer.print("Processors to use for ANOVA: " + str(proc_to_use))
    if proc_to_use == 1:
        res = [anova_filter_one_feature_checked(x=view[view.columns[[i]]], y=outcome, p_val=p_val, verbose=verbose)
               for i in range(n_cols)]
    else:
        inputs = (OneColInput(x=view[view.columns[[i]]], y=outcome, p_val=p_val, verbose=verbose)
                  for i in range(n_cols))
        with ProcessPoolExecutor(max_workers=proc_to_use) as workers_pool:
            res = workers_pool.map(
                anova_filter_one_col_input_checked, inputs, chunksize=16)
            res = list(res)
    if verbose:
        printer.print_variable("ANOVA filter execution time", pretty_duration(time.time() - start_time))
        nan_s = nan_slice(select_cols_by_mask(view, res))
        printer.print("NaN slice after filter:")
        printer.print(str(nan_s))
    return res


def filter_anova(view, outcome, p_val: float = DEFAULT_P_VAL, n_proc: int = 1, printer: Printer = DEFAULT_PRINTER):
    res = view.copy()
    mask = filter_anova_mask(view=view, outcome=outcome, p_val=p_val, n_proc=n_proc, printer=printer)
    res = res.loc[:, mask]
    return res
