import itertools
from collections import Iterable, Sized, Sequence

import numpy as np

from util.dataframes import create_from_labelled_lists
from util.named import Named


def str_in_lines(li: Iterable) -> str:
    res = ""
    for i in li:
        res += str(i) + "\n"
    return res


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


def names(it: Iterable[Named]) -> list[str]:
    return [i.name() for i in it]


def stable_uniques(x: Iterable) -> list:
    """Returns list of unique elements preserving order of first encounter."""
    return list(dict.fromkeys(x))


def sum_constant(x: Iterable, c: float) -> np.ndarray:
    ar = np.asarray(x)
    return ar + c


def transpose(x: Iterable[Iterable]) -> list[list]:
    """From list of rows to list of columns or vice versa."""
    return list(map(list, zip(*x)))


def to_common_labels(lists: [list]) -> [list]:
    df = create_from_labelled_lists(lists=lists)
    return df.values.tolist()


def equal(a: Iterable, b: Iterable) -> bool:
    """Works with any pair of objects with a working iter function."""
    a_iter = iter(a)
    if a is b:  # Done after the iter call so that we get an exception if the object is not iterable.
        return True
    b_iter = iter(b)
    if isinstance(a, Sized) and isinstance(b, Sized):
        if len(a) != len(b):
            return False
    while True:
        try:
            a_next = next(a_iter)
        except StopIteration:
            try:
                next(b_iter)
                return False  # a is shorter
            except StopIteration:
                return True  # Same length
        try:
            b_next = next(b_iter)
        except StopIteration:
            return False  # b is shorter
        if not a_next == b_next:  # Since it is not guaranteed that != is the same as not ==
            return False


def flatten_iterable_of_iterable(x: Iterable[Iterable]) -> list:
    return list(itertools.chain.from_iterable(x))


def binary_search_iterative(array: Sequence, element):
    """Returns the position of the element, or ValueError if not found."""
    start = 0
    end = len(array)

    while start <= end:
        mid = (start + end) // 2

        if element == array[mid]:
            return mid

        if element < array[mid]:
            end = mid - 1
        else:
            start = mid + 1
    raise ValueError()


def sort_permutation(s: Sequence, reverse: bool = False) -> list[int]:
    """From lowest to highest unless reverse is True."""
    return sorted(range(len(s)), key=lambda k: s[k], reverse=reverse)


def count_nonzero(s: Sequence[float]) -> int:
    return sum([x != 0.0 for x in s])
