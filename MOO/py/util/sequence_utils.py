import itertools
from collections.abc import Iterable, Sequence

from util.dataframes import create_from_labelled_lists


def sequence_to_string(li, compact=False) -> str:
    res = "["
    first = True
    for i in li:
        if first:
            res += str(i)
            first = False
        else:
            if compact:
                res += ","
            else:
                res += ", "
            res += str(i)
    return res + "]"


def str_in_lines(li: Iterable) -> str:
    res = ""
    for i in li:
        res += str(i) + "\n"
    return res


def sort_permutation(s: Sequence, reverse: bool = False) -> list[int]:
    """From lowest to highest unless reverse is True."""
    return sorted(range(len(s)), key=lambda k: s[k], reverse=reverse)


def tuple_to_string(tup):
    res = "("
    first = True
    for i in tup:
        if first:
            res += str(i)
            first = False
        else:
            res += ", " + str(i)
    return res + ")"


def stable_uniques(x: Iterable) -> list:
    """Returns list of unique elements preserving order of first encounter."""
    return list(dict.fromkeys(x))


def to_common_labels(lists: [list]) -> [list]:
    df = create_from_labelled_lists(lists=lists)
    return df.values.tolist()


def flatten_iterable_of_iterable(x: Iterable[Iterable]) -> list:
    return list(itertools.chain.from_iterable(x))


def transpose(x: Iterable[Iterable]) -> list[list]:
    """From list of rows to list of columns or vice versa."""
    return list(map(list, zip(*x)))
