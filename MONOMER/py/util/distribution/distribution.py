from __future__ import annotations
from abc import ABC
from random import random
from typing import Optional, Sequence

from util.list_math import list_div
from util.sequence_utils import sequence_to_string, sort_permutation
from util.sparse_bool_list_by_set import SparseBoolList, SparseBoolListBySet
from util.summer import KahanSummer


class Distribution(Sequence[float], ABC):
    """A sequence of non-negative floats that sum to 1."""

    def extract(self) -> int:
        """Uses package random.
        This basic implementation can be overridden for speed
        by memorizing the cumulative sums and using binary search."""
        r = random()
        summer = KahanSummer()
        i = 0
        summer.add(self[i])
        last_i = len(self) - 1
        while r >= summer.get_sum() and i < last_i:
            # Second condition is to handle cases when the sum is of all probabilities is slightly less than 1
            # due to numerical approximation
            i += 1
            summer.add(self[i])
        return i

    def cumulative(self) -> Sequence[float]:
        """Value at position i is sum up to i included."""
        n = len(self)
        res = [0.0]*n
        summer = KahanSummer()
        for i in range(n):
            summer.add(self[i])
            res[i] = min(1.0, summer.get_sum())
        return res

    def as_cached(self):
        return CachedDistribution(inner=self)

    def __str__(self):
        return sequence_to_string(self)

    def nonzero(self) -> SparseBoolList:
        res = SparseBoolListBySet(min_size=len(self))
        for i, x in enumerate(self):
            if x > 0.0:
                res.set_true(i)
        return res

    def nonzero_num(self) -> int:
        return sum(self.nonzero())

    def focus(self, max_elems: int) -> Distribution:
        """All elements except the highest max_elems are set to zero, and the result is scaled to sum to 1."""
        if max_elems >= len(self):
            return self
        else:
            res_indices = sort_permutation(s=self)[0:max_elems]
            res_seq = [0.0]*len(self)
            for i in res_indices:
                res_seq[i] = self[i]
            return ConcreteDistribution(probs=res_seq)

    def is_uniform(self) -> bool:
        if len(self) == 0:
            return True
        else:
            val = self[0]
            for x in self:
                if x != val:
                    return False
            return True


class CachedDistribution(Distribution):
    __inner: Distribution
    __cumulative: Optional[Sequence[float]]

    def __init__(self, inner: Distribution):
        self.__inner = inner
        self.__cumulative = None

    def __getitem__(self, i: int) -> float:
        return self.__inner[i]

    def __len__(self) -> int:
        return len(self.__inner)

    def cumulative(self) -> Sequence[float]:
        if self.__cumulative is None:
            self.__cumulative = self.__inner.cumulative()
        return self.__cumulative

    def as_cached(self):
        return self

    def extract(self) -> int:
        """Uses package random. Uses binary search."""
        r = random()
        cum = self.cumulative()
        start = 0
        end = len(cum) - 1
        while start < end:
            mid = (start + end) // 2
            if r <= cum[mid]:
                end = mid
            else:
                start = mid + 1
        return start

    def extract_old(self) -> int:
        """Uses package random. Old version kept for testing purposes. Binary search is faster."""
        r = random()
        i = 0
        cum = self.cumulative()
        last_i = len(self) - 1
        while r >= cum[i] and i < last_i:
            # Second condition is to handle cases when the sum is of all probabilities is slightly less than 1
            # due to numerical approximation
            i += 1
        return i


class ConcreteDistribution(Distribution):
    __probs: Sequence[float]

    def __init__(self, probs: Sequence[float]):
        """probs are scaled to sum to 1 if they are not already.
        If all probs are zero we create a uniform distribution."""
        if isinstance(probs, Distribution):
            self.__probs = probs
        else:
            p_sum = KahanSummer.sum(probs)
            if p_sum == 0.0:  # If all probs are zero we create a uniform distribution.
                n_probs = len(probs)
                self.__probs = [1.0/n_probs]*n_probs
            else:
                self.__probs = list_div(probs, KahanSummer.sum(probs))

    def __getitem__(self, i: int) -> float:
        return self.__probs[i]

    def __len__(self) -> int:
        return len(self.__probs)
