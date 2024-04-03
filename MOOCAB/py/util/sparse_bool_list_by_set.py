from abc import ABC
from typing import Iterable

import numpy
import numpy as np
from numpy import ndarray

from util.math import list_math
from util.list_like import ListLike
from util.math.list_math import top_k_positions
from util.math.summer import KahanSummer
from sortedcontainers import SortedSet


class SparseBoolList(ListLike, ABC):

    def __add_with_sparse(self, sparse):
        res = numpy.zeros(len(self), int)
        for a in self.true_positions():
            res[a] = 1
        for a in sparse.true_positions():
            res[a] += 1
        return res

    def add(self, li: ListLike):
        if len(self) != len(li):
            raise ValueError("Adding lists of different sizes.")
        if isinstance(li, SparseBoolList):
            return self.__add_with_sparse(li)
        else:
            res = list(li)
            for a in self.true_positions():
                res[a] += 1
            return res

    def __dot_product_with_sparse(self, sparse):
        res = 0
        if len(self.true_positions()) >= len(sparse.true_positions()):
            sparse_big = self
            sparse_small = sparse
        else:
            sparse_big = sparse
            sparse_small = self
        for a in sparse_small.true_positions():
            if a in sparse_big.true_positions():
                res += 1
        return res

    def dot_product(self, li: ListLike, summer_class=KahanSummer):
        if len(self) != len(li):
            raise ValueError("Dot product with lists of different sizes.")
        if isinstance(li, SparseBoolList):
            return self.__dot_product_with_sparse(li)
        else:
            summer = summer_class.create()
            for e in self.true_positions():
                summer.add(li[e])
        return summer.get_sum()

    def __eq__(self, other: ListLike):
        if self is other:
            return True
        le = len(self)
        if le != len(other):
            return False
        if isinstance(other, SparseBoolList):
            return self.true_positions() == other.true_positions()
        for i in range(le):
            if other[i]:
                if not self[i]:
                    return False
            else:
                if self[i]:
                    return False
        return True

    def __hash__(self):
        return hash((frozenset(self.true_positions()), len(self)))


class SparseBoolListBySet(SparseBoolList):
    __s: SortedSet[int]
    __size: int

    def __init__(self, seq=(), min_size=0):
        """Length is set to max between the length of seq and passed min_size.
        Unspecified positions are filled with zeros."""
        if isinstance(seq, SparseBoolListBySet):
            self.__s = seq.true_positions().copy()
            self.__size = max(len(seq), min_size)
        else:
            self.__s = SortedSet()
            self.__size = 0
            self.extend(seq)
            self.__size = max(self.__size, min_size)

    def true_positions(self) -> SortedSet[int]:
        return self.__s

    def __len__(self):
        return self.__size

    def __get_one_item(self, key: int) -> bool:
        if key < 0 or key >= self.__size:
            raise ValueError()
        return key in self.__s

    def __getitem__(self, key):
        """Supports integer key or slice key."""
        if isinstance(key, (int, np.integer)):
            return self.__get_one_item(key=key)
        if isinstance(key, slice):
            indices = range(*key.indices(self.__size))
            return [self.__get_one_item(i) for i in indices]  # TODO Could return a sparse list for higher efficiency.
        raise TypeError()

    def __setitem__(self, key, value):
        if not isinstance(key, (int, np.integer)):
            raise TypeError()
        if key < 0 or key >= self.__size:
            raise ValueError()
        if value:
            self.__s.add(key)
        else:
            self.__s.discard(key)

    def set_true(self, key):
        if not isinstance(key, (int, np.integer)):
            raise TypeError("Wrong key: " + str(key) + " of type " + str(type(key)))
        if key < 0 or key >= self.__size:
            raise ValueError("passed key: " + str(key) + ", size: " + str(self.__size))
        self.__s.add(key)

    def set_all_true(self, keys):
        for k in keys:
            self.set_true(k)

    def __extend_one(self, b):
        if b:
            self.__s.add(self.__size)
        self.__size += 1

    def __extend_with_sparse(self, sparse: SparseBoolList):
        self_size = self.__size
        self.__size = self_size + len(sparse)
        for e in sparse.true_positions():
            self.__s.add(e+self_size)

    def extend(self, it: Iterable):
        if isinstance(it, SparseBoolList):
            self.__extend_with_sparse(it)
        else:
            for i in it:
                self.__extend_one(i)

    def to_numpy(self) -> ndarray:
        result = np.full(shape=len(self), fill_value=False)
        for i in self.__s:
            result[i] = True
        return result

    def sum(self):
        return len(self.__s)

    def append(self, value):
        self.__extend_one(value)

    def __str__(self):
        ret_string = "SparseBoolListBySet of size " + str(self.__size) + "\n"
        ret_string += str(len(self.__s)) + " True positions:\n"
        ret_string += str(self.__s) + "\n"
        return ret_string

    def __copy__(self):
        return SparseBoolListBySet(seq=self)

    def __deepcopy__(self, memodict={}):
        return SparseBoolListBySet(seq=self)

    def extend_to(self, min_size: int):
        """Extends with false values if necessary to get to at least this size."""
        self.__size = max(self.__size, min_size)


def chain(lists):
    res = SparseBoolListBySet()
    for li in lists:
        res.extend(li)
    return res


def dot_product(list_a: ListLike, list_b: ListLike, summer_class=KahanSummer):
    if isinstance(list_a, SparseBoolList):
        return list_a.dot_product(list_b, summer_class)
    if isinstance(list_b, SparseBoolList):
        return list_b.dot_product(list_a, summer_class)
    return list_math.dot_product(list_a, list_b, summer_class=summer_class)


def smart_sum(listlike: ListLike, summer_class=KahanSummer):
    if isinstance(listlike, SparseBoolList):
        return listlike.sum()
    else:
        return summer_class.sum(listlike)


def add(list_a: ListLike, list_b: ListLike):
    if len(list_a) != len(list_b):
        raise ValueError("Adding lists of different length.\n" +
                         "len(list_a): " + str(len(list_a)) + "\n" +
                         "len(list_b): " + str(len(list_b)) + "\n")
    if isinstance(list_a, SparseBoolList):
        return list_a.add(list_b)
    if isinstance(list_b, SparseBoolList):
        return list_b.add(list_a)
    return list_math.list_add(list_a, list_b)


def equal(list_a: ListLike, list_b: ListLike) -> bool:
    if list_a is list_b:
        return True
    if len(list_a) != len(list_b):
        return False
    if isinstance(list_a, SparseBoolList):
        return list_a.__eq__(list_b)
    if isinstance(list_b, SparseBoolList):
        return list_b.__eq__(list_a)
    return list_a == list_b


def jaccard_score(y_true: SparseBoolListBySet, y_pred: SparseBoolListBySet, zero_division=1.0) -> float:
    intersection = 0
    union_size = y_true.sum() + y_pred.sum()
    for i in y_true.true_positions():
        if y_pred[i]:
            intersection += 1
            union_size -= 1
    if union_size == 0:
        return zero_division
    else:
        return intersection / union_size


def union_of_pair_sparse(list_a: SparseBoolList, list_b: SparseBoolList) -> SparseBoolListBySet:
    """If lengths are different, the result has the bigger length."""
    if sum(list_a) < sum(list_b):
        small = list_a
        big = list_b
    else:
        small = list_b
        big = list_a
    res_len = max(len(list_a), len(list_b))
    res = SparseBoolListBySet(seq=big, min_size=res_len)
    for tp in small.true_positions():
        res[tp] = True
    return res


def union_of_pair_sparse_a(list_a: SparseBoolList, list_b: ListLike) -> ListLike:
    """If lengths are different, the result has the bigger length."""
    if isinstance(list_b, SparseBoolList):
        return union_of_pair_sparse(list_a, list_b)
    else:
        res = list(list_b)
        res.extend([False] * max(0, (len(list_a) - len(list_b))))
        for tp in list_a.true_positions():
            res[tp] = True
        return res


def union_of_pair(list_a: ListLike, list_b: ListLike) -> ListLike:
    """If lengths are different, the result has the bigger length."""
    if isinstance(list_a, SparseBoolList):
        return union_of_pair_sparse_a(list_a, list_b)
    if isinstance(list_b, SparseBoolList):
        return union_of_pair_sparse_a(list_b, list_a)
    len_a = len(list_a)
    len_b = len(list_b)
    if len_a >= len_b:
        big = list_a
        small = list_b
        len_small = len_b
    else:
        big = list_b
        small = list_a
        len_small = len_a
    res = list(big)
    for i in range(len_small):
        if small[i]:
            res[i] = True
    return res


def union_sparse(sets: Iterable[SparseBoolList]) -> SparseBoolListBySet:
    """If lengths are different, the result has the bigger length."""
    res = SparseBoolListBySet()
    for s in sets:
        res = union_of_pair_sparse(res, s)
    return res


def union(sets: Iterable[ListLike]) -> SparseBoolListBySet:
    """If lengths are different, the result has the bigger length."""
    res = SparseBoolListBySet()
    for s in sets:
        len_s = len(s)
        res.extend_to(len_s)
        if isinstance(s, ListLike):
            res.set_all_true(s.true_positions())
        else:
            for i in range(len_s):  # This part works also with lists that are not ListLike.
                if s[i]:
                    res.set_true(i)
    return res


def top_k_mask(elems: ListLike, k: int, postconditions: bool = False) -> SparseBoolListBySet:
    """Returns a 0/1 mask with 1s in the top k positions. Uses stable sorting."""
    positions = top_k_positions(elems, k)
    res = SparseBoolListBySet(min_size=len(elems))
    res.set_all_true(positions)
    if postconditions:
        ct = 0
        for r in res:
            if r == 1:
                ct += 1
            if r != 0 and r != 1:
                raise AssertionError()
        if ct != k:
            raise AssertionError()
    return res
