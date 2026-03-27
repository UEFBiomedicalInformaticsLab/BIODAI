import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

from os import devnull


def has_method(obj, name):
    return callable(getattr(obj, name, None))


class IllegalStateError(RuntimeError):
    pass


class PlannedUnreachableCodeError(RuntimeError):
    pass


def files_by_extesion(dir_name: str, extension: str) -> Sequence[str]:
    """Extension passed is without .
    Strings returned include the directory."""
    res = []
    for file in os.listdir(dir_name):
        if file.endswith("." + extension):
            res.append(os.path.join(dir_name, file))
    return res


def change_extension(file: str, new_ext: str) -> str:
    base = os.path.splitext(file)
    return base[0] + '.' + new_ext


def mean_of_dataframes(dfs):
    """Returns a dataframe with the mean of passed dataframes cell-wise.
    Passed dataframes must have the same columns."""
    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    return df_means


def try_make_file(filename) -> bool:
    """Atomically tests if a file exists and if not creates it. Returns true if it created a new file.
    Note: it is not clear from the documentation of open if it is guaranteed to be atomic on all platforms."""
    try:
        with open(file=filename, mode="x") as _:
            return True
    except FileExistsError:
        return False


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    From https://stackoverflow.com/a/33532498/992687"""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def is_sequence_not_string(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def null_showwarning(message, category=UserWarning, filename='', lineno=-1):
    pass


def same_len(sequences: Sequence[Sequence]) -> bool:
    n_sequences = len(sequences)
    if n_sequences == 0:
        return True
    size = len(sequences[0])
    for i in range(1, n_sequences):
        if len(sequences[i]) != size:
            return False
    return True


def bound(x, min_x, max_x):
    if x <= min_x:
        return min_x
    else:
        if x >= max_x:
            return max_x
        else:
            return x
