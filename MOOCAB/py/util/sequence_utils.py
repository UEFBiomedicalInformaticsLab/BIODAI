import itertools
from collections.abc import Iterable, Sized, Sequence
from typing import Any

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


def sequence_to_string(li: Iterable, compact=False, separator=",", brackets: bool = True) -> str:
    res = ""
    if brackets:
        res += "["
    first = True
    for i in li:
        if first:
            res += str(i)
            first = False
        else:
            if compact:
                res += separator
            else:
                res += separator + " "
            res += str(i)
    if brackets:
        res += "]"
    return res


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


def same_len(a: Iterable, b: Iterable) -> bool:
    """Works with any pair of objects with a working iter function.,
    Uses the len method if possible, otherwise iterates."""
    a_iter = iter(a)
    if a is b:  # Done after the iter call so that we get an exception if the object is not iterable.
        return True
    b_iter = iter(b)
    if isinstance(a, Sized) and isinstance(b, Sized):
        return len(a) == len(b)
    else:
        while True:
            try:
                next(a_iter)
            except StopIteration:
                try:
                    next(b_iter)
                    return False  # a is shorter
                except StopIteration:
                    return True  # Same length
            try:
                next(b_iter)
            except StopIteration:
                return False  # b is shorter


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


def ordered_counter(elems: Iterable[Any]) -> dict[int, Any]:
    """Counts keeping the order of first encounter."""
    counter = {}
    for e in elems:
        counter[e] = counter.get(e, 0)+1
    return counter


def filter_by_booleans(data: Iterable, selectors: Iterable[bool]) -> list:
    return list(itertools.compress(data, selectors))


def clean_redundant_subsequences(data: Sequence[Sequence]) -> list[Sequence]:
    """If a sequence has all elements contained in another sequence, it is removed."""
    data_len = len(data)
    sets = [set(d) for d in data]
    to_keep = [True]*data_len
    for i in range(data_len):
        i_set = sets[i]
        for j in range(data_len):
            if i != j:
                j_set = sets[j]
                if i_set.issubset(j_set):
                    if i < j:
                        to_keep[i] = False
                    else:  # We do not want to remove both i and j sets if they are equal.
                        if not j_set.issubset(i_set):
                            to_keep[i] = False
    res = []
    for i in range(data_len):
        if to_keep[i]:
            res.append(data[i])
    return res


def select_by_indices(data: Sequence, indices: Iterable[int]) -> list:
    """If the sequence is a DataFrame, the indexing will be by row names."""
    try:
        return [data[i] for i in indices]
    except KeyError as e:
        raise KeyError("KeyError when accessing an element of " + str(data) + "\n" + str(e))


def list_of_empty_lists(n: int) -> list[list]:
    return [[] for _ in range(n)]


def sort_both_by_first(seq1: Sequence, seq2: Sequence) -> tuple[Sequence, Sequence]:
    """New sequences are created containing the original elements."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences should have the same length.")
    if len(seq1) == 0:
        return (), ()
    return zip(*sorted(zip(seq1, seq2), key=lambda x: x[0]))


def reverse(seq: Sequence) -> list:
    last_index = len(seq) - 1
    return [seq[i] for i in range(last_index, -1, -1)]
