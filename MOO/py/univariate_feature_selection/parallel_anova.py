import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List

from numpy import ravel, var
from sklearn import feature_selection
from util.dataframes import nan_slice, select_cols_by_mask
from util.utils import pretty_duration


def anova_filter_one_feature(x, y, p_val=0.05, verbose: bool = True, ignore_warns: bool = True) -> bool:
    try:
        if ignore_warns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                anova_res = feature_selection.f_classif(X=x, y=y)
        else:
            anova_res = feature_selection.f_classif(X=x, y=y)
        p_vals = anova_res[1]
        res = p_vals[0] < p_val
    except BaseException as e:
        if verbose:
            print("Exception encountered in ANOVA:")
            print(e)
            print("x:")
            print(x)
            rx = ravel(x)
            print("Variance: " + str(var(rx)))
            print("Filter will return False.")
        res = False
    return res


def anova_filter_one_feature_checked(x, y, p_val=0.05, verbose: bool = True, ignore_warns: bool = True) -> bool:
    """Checks for NaN and zero variance otherwise f_classif does not work.
    Returns False if feature is discarded."""
    if x.isna().any().any():
        return False
    if not (x.var() > 0.0).all():
        return False
    return anova_filter_one_feature(x=x, y=y, p_val=p_val, verbose=verbose, ignore_warns=ignore_warns)


class OneColInput:

    def __init__(self, x, y, p_val, verbose):
        self.x = x
        self.y = y
        self.p_val = p_val
        self.verbose = verbose


def anova_filter_one_col_input_checked(col_input: OneColInput) -> bool:
    return anova_filter_one_feature_checked(
        x=col_input.x, y=col_input.y, p_val=col_input.p_val,
        verbose=col_input.verbose)


def filter_anova_mask(view, outcome, p_val: float = 0.05, n_proc: int = 1, verbose: bool = False) -> List[bool]:
    """Checks also for NaN and zero variance otherwise f_classif does not work."""
    n_cols = len(view.columns)
    cpu_count = multiprocessing.cpu_count()
    proc_to_use = max(1, min(n_proc, cpu_count, n_cols))
    start_time = None
    if verbose:
        start_time = time.time()
        print("Processors to use for ANOVA: " + str(proc_to_use))
    if proc_to_use == 1:
        res = [anova_filter_one_feature_checked(x=view[view.columns[[i]]], y=outcome, p_val=p_val, verbose=verbose)
               for i in range(n_cols)]
    else:
        inputs = [OneColInput(x=view[view.columns[[i]]], y=outcome, p_val=p_val, verbose=verbose)
                  for i in range(n_cols)]
        with ProcessPoolExecutor(max_workers=proc_to_use) as workers_pool:
            res = workers_pool.map(
                anova_filter_one_col_input_checked, inputs, chunksize=16)
            res = list(res)
    if verbose:
        print("ANOVA filter execution time: ", pretty_duration(time.time() - start_time))
        nan_s = nan_slice(select_cols_by_mask(view, res))
        print("NaN slice after filter:")
        print(str(nan_s))
    return res


def filter_anova(view, outcome, p_val: float = 0.05, n_proc: int = 1):
    res = view.copy()
    mask = filter_anova_mask(view=view, outcome=outcome, p_val=p_val, n_proc=n_proc)
    res = res.loc[:, mask]
    return res
