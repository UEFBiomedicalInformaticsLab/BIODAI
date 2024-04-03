import os
import sys
from collections.abc import Sequence, Iterable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
import pandas as pd
from numpy import unique

from util.named import Named
from util.sequence_utils import sequence_to_string, transpose
from os import devnull

from util.math.summer import KahanSummer


def has_method(obj, name):
    return callable(getattr(obj, name, None))


def pretty_duration(duration_seconds) -> str:
    d = int(duration_seconds)
    days = d // 86400
    hours = d % 86400 // 3600
    mins = d % 3600 // 60
    secs = d % 60
    return '{:02d}-{:02d}:{:02d}:{:02d}'.format(days, hours, mins, secs)


def name_str(x):
    if isinstance(x, Named):
        return x.name()
    else:
        return str(x)


def names(elems: Sequence) -> [str]:
    strings = []
    for e in elems:
        strings.append(name_str(e))
    return strings


def names_str(elems: Sequence) -> str:
    return sequence_to_string(names(elems))


def name_value(name: str, value):
    if isinstance(name, Named):
        str_name = name.name()
    else:
        str_name = str(name)
    if isinstance(value, Named):
        str_val = value.name()
    elif isinstance(value, str):
        str_val = value
    elif isinstance(value, Sequence):
        str_val = names_str(value)
    else:
        str_val = str(value)
    return str_name + " = " + str_val


def feature_names(column_names: list[str], active_feature_positions: set[int]) -> list[str]:
    res = []
    for a in active_feature_positions:
        res.append(column_names[a])
    return res


def feature_names_from_collapsed_views(collapsed_views, active_feature_positions: set[int]) -> list[str]:
    return feature_names(column_names=collapsed_views.columns, active_feature_positions=active_feature_positions)


def automatic_to_string(x) -> str:
    return str(vars(x))


class IllegalStateError(RuntimeError):
    pass


class PlannedUnreachableCodeError(RuntimeError):
    pass


def files_by_extesion(dir_name: str, extension: str) -> [str]:
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


def dict_select(old_dict: dict, keys: Iterable) -> dict:
    """Raises exception if a key is not present."""
    return {k: old_dict[k] for k in keys}


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


def sorted_dict(d: dict) -> dict:
    """Sorted by keys."""
    res = {}
    for key, value in sorted(d.items(), key=lambda x: x[0]):
        res[key] = value
    return res


def str_dict(d: dict, in_lines: bool = False) -> str:
    res = ""
    if not in_lines:
        res += "{"
    first = True
    for key in d:
        value = d[key]
        if first:
            first = False
        else:
            if in_lines:
                res += "\n"
            else:
                res += ", "
        res += str(key) + ": " + str(value)
    if not in_lines:
        res += "}"
    return res


def str_sorted_dict(d: dict, in_lines: bool = False) -> str:
    return str_dict(sorted_dict(d), in_lines=in_lines)


def str_paste(parts: Sequence[str], separator: str) -> str:
    res = ""
    for i in range(len(parts)):
        if i > 0:
            res = res + separator
        res = res + str(parts[i])
    return res


def ceil_division(num: int, den: int) -> int:
    """Integer division that rounds up instead of rounding down."""
    return -(-num // den)


def dict_sort_by_value(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1]))


def mean_of_dicts(dicts: Sequence[dict[Any, float]]) -> dict[Any, float]:
    """Keys not present in a dict are considered zero for the mean."""
    to_average = {}
    for d in dicts:
        for k in d:
            if k in to_average:
                to_average[k].append(d[k])
            else:
                to_average[k] = [d[k]]
    res = {}
    num_elems = float(len(dicts))
    for k in to_average:
        res[k] = KahanSummer.sum(to_average[k]) / num_elems
    return res


def string_from_selected(parts: Sequence[str], selected: Sequence[bool], separator: str = " ") -> str:
    res = ""
    for p, s in zip(parts, selected):
        if s:
            if res != "":
                res += separator
            res += p
    return res


def names_by_differences(object_features: Sequence[Sequence[str]], separator: str = " ") -> Sequence[str]:
    """The outer sequence is for objects, the inner sequences are the features.
    Each object must have the same number of features.
    Returns a string for each object, avoiding the features that all objects have in common."""
    n_obj = len(object_features)
    if n_obj == 0:
        return []
    transposed = transpose(object_features)
    to_use = []
    for f in transposed:
        if len(unique(f)) > 1:
            to_use.append(True)
        else:
            to_use.append(False)
    return [string_from_selected(parts=o, selected=to_use, separator=separator) for o in object_features]
