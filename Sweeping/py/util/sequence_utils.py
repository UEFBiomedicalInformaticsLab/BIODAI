import itertools
import math
import warnings
from collections.abc import Iterable, Sized, Iterator
from typing import Any, Sequence, Union, TypeVar

import numpy as np
from sortedcontainers import SortedSet

from util.dataframe.dataframe_creators import create_from_labelled_lists
from util.list_like import ListLike
from numpy import ndarray


def stable_uniques(x: Iterable) -> list:
    """Returns list of unique elements preserving order of first encounter."""
    return list(dict.fromkeys(x))


def sum_constant(x: Iterable, c: float) -> np.ndarray:
    ar = np.asarray(x)
    return ar + c


def transpose(x: Iterable[Iterable]) -> list[list]:
    """From list of rows to list of columns or vice versa."""
    return list(map(list, zip(*x)))


def to_common_labels(lists: Sequence[list]) -> Sequence[list]:
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


def equal_iterables(a: Iterable, b: Iterable) -> bool:
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


def sort_permutation(s: Sequence, in_reverse: bool = False) -> list[int]:
    """From lowest to highest unless in_reverse is True.
    E.g. sort_permutation(s=[3,1,5,2,7]) -> [1, 3, 0, 2, 4]"""
    return sorted(range(len(s)), key=lambda k: s[k], reverse=in_reverse)


def ranks(s: Sequence) -> ndarray:
    """ranks(s=[3,1,5,2,7]) -> [2,0,3,1,4]
    ranks(s=[5,3,1,5,2,7,2]) -> [4,3,0,5,1,6,2]"""
    return np.argsort(np.argsort(s, kind="stable"), kind="stable")


def count_nonzero(s: Sequence[float]) -> int:
    return sum([x != 0.0 for x in s])


def true_positions(s: Iterable) -> Sequence[int]:
    """Optimizes for ListLike if possible."""
    if isinstance(s, ListLike):
        return s.true_positions()
    res = []
    for i, e in enumerate(s):
        if e:
            res.append(i)
    return res


def true_positions_sorted_set(s: Iterable) -> SortedSet[int]:
    """Optimizes for ListLike if possible."""
    if isinstance(s, ListLike):
        return s.true_positions()
    res = SortedSet()
    for i, e in enumerate(s):
        if e:
            res.add(i)
    return res


def ordered_counter(elems: Iterable[Any]) -> dict[int, Any]:
    """Counts keeping the order of first encounter."""
    counter = {}
    for e in elems:
        counter[e] = counter.get(e, 0)+1
    return counter


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


def as_list(it: Iterable) -> list:
    if isinstance(it, list):
        return it
    else:
        return list(it)


def strictly_increasing(elems: Sequence) -> bool:
    return all(x<y for x, y in zip(elems, elems[1:]))


def strictly_decreasing(elems: Sequence) -> bool:
    return all(x>y for x, y in zip(elems, elems[1:]))


def max_positions(elems: Sequence) -> set[int]:
    """The positions of the elements that have the highest value."""
    highest_value = float('-inf')
    positions = {}
    for index, value in enumerate(elems):
        if value > highest_value:
            highest_value = value
            positions = {index}
        elif value == highest_value:
            positions.add(index)
    return positions


class NotNanIterator(Iterator[np.number]):
    __inner: Iterator[np.number]

    def __init__(self, inner: Iterator[np.number]):
        self.__inner = inner

    def __next__(self):
        res = math.nan
        while math.isnan(res):
          res = next(self.__inner)
        return res


class NotNanIterable(Iterable[np.number], Sized):
    """Size is not cached because the inner iterable might change."""
    __inner: Iterable[np.number]

    def __init__(self, inner: Iterable[np.number]):
        self.__inner = inner

    def __iter__(self) -> Iterator[np.number]:
        return NotNanIterator(inner=iter(self.__inner))

    def __len__(self) -> int:
        tot = 0
        for x in self.__inner:
            if not math.isnan(x):
                tot += 1
        return tot


def seq_intersection(sequences: Sequence[Sequence]) -> Sequence:
    n_seq = len(sequences)
    if n_seq == 0:
        raise ValueError()
    else:
        res = sequences[0]
        for i in range(1, n_seq):
            res = [r for r in res if r in sequences[i]]
        return res


def safe_nanmin(seq: Union[Sequence,np.array]) -> float:
    """returns the min ignoring NaNs. If all elements are NaN or seq is empty, returns NaN."""
    arr = np.asarray(seq)
    if arr.size == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(arr)


def safe_nanmax(seq: Union[Sequence,np.array]) -> float:
    """returns the max ignoring NaNs. If all elements are NaN or seq is empty, returns NaN."""
    arr = np.asarray(seq)
    if arr.size == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(arr)


T = TypeVar("T")

def concat(a: Sequence[T], b: Sequence[T]) -> list[T]:
    return [*a, *b]
