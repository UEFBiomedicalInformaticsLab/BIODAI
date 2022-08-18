import os
from collections.abc import Sequence

import pandas as pd

from util.named import Named
from util.sequence_utils import sequence_to_string


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


class IllegalStateError(RuntimeError):
    pass


def try_make_file(filename) -> bool:
    """Atomically tests if a file exists and if not creates it. Returns true if it created a new file.
    Note: it is not clear from the documentation of open if it is guaranteed to be atomic on all platforms."""
    try:
        with open(file=filename, mode="x") as _:
            return True
    except FileExistsError:
        return False


def dict_select(old_dict: dict, keys) -> dict:
    """Raises exception if a key is not present."""
    return {k: old_dict[k] for k in keys}


def same_len(sequences: Sequence[Sequence]) -> bool:
    n_sequences = len(sequences)
    if n_sequences == 0:
        return True
    size = len(sequences[0])
    for i in range(1, n_sequences):
        if len(sequences[i]) != size:
            return False
    return True


def mean_of_dataframes(dfs):
    """Returns a dataframe with the mean of passed dataframes cell-wise.
    Passed dataframes must have the same columns."""
    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    return df_means


def change_extension(file: str, new_ext: str) -> str:
    base = os.path.splitext(file)
    return base[0] + '.' + new_ext


def str_paste(parts: Sequence[str], separator: str) -> str:
    res = ""
    for i in range(len(parts)):
        if i > 0:
            res = res + separator
        res = res + str(parts[i])
    return res


def is_sequence_not_string(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)


def ceil_division(num: int, den: int) -> int:
    """Integer division that rounds up instead of rounding down."""
    return -(-num // den)
