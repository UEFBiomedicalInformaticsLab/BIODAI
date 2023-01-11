import itertools
from collections import Sequence
from itertools import compress
from typing import List

import numpy as np
from numpy.core import umath

from util.summer import KahanSummer


def dot_product(list_a: list, list_b: list, summer_class=KahanSummer):
    return summer_class.sum([x*y for x, y in zip(list_a, list_b)])


def list_add(list_a: list, list_b: list):
    if len(list_a) != len(list_b):
        raise ValueError("Adding lists of different length.")
    return [x + y for (x, y) in zip(list_a, list_b)]


def list_subtract(list_a: list, list_b: list):
    return [x - y for (x, y) in zip(list_a, list_b)]


def n_true_in_common(list_a: list, list_b: list):
    return sum([(bool(x) and bool(y)) for (x, y) in zip(list_a, list_b)])


def jaccard(list_a: list, list_b: list):
    len_a = len(list_a)
    if len_a != len(list_b):
        raise ValueError()
    num = 0
    den = 0
    for i in range(len_a):
        true_a = list_a[i] != 0
        true_b = list_b[i] != 0
        if true_a or true_b:
            den += 1
            if true_a and true_b:
                num += 1
    return num / den


def dice(list_a: list, list_b: list):
    len_a = len(list_a)
    if len_a != len(list_b):
        raise ValueError()
    num = 0
    den = 0
    for i in range(len_a):
        true_a = list_a[i] != 0
        true_b = list_b[i] != 0
        if true_a and true_b:
            num += 2
        if true_a:
            den += 1
        if true_b:
            den += 1
    return num / den


def list_tot_abs_difference(list_a: list, list_b: list, summer_class=KahanSummer):
    return summer_class.sum([abs(x - y) for (x, y) in zip(list_a, list_b)])


def list_div(li: Sequence, d: float):
    return [x / d for x in li]


def list_multiply(list_a, list_b):
    return [a*b for a, b in zip(list_a, list_b)]


def list_not(li: List[bool]) -> List[bool]:
    return [not elem for elem in li]


def list_and(list_a: [bool], list_b: [bool]):
    return [a and b for a, b in zip(list_a, list_b)]


def list_or(list_a: List[bool], list_b: List[bool]):
    return [a or b for a, b in zip(list_a, list_b)]


def mean_all_vs_others(elems: Sequence, measure_function):
    n = len(elems)
    if n < 2:
        raise ValueError("At least two elements are needed to avoid division by zero.")
    summation = KahanSummer()
    for j in range(n):
        for k in range(j+1, n):
            summation.add(measure_function(elems[j], elems[k]))
    denominator = ((n*n)-n)/2
    res = summation.get_sum()/denominator
    return res


def indices_of_true(elems: list) -> list:
    return list(compress(range(len(elems)), elems))


def num_of_true(elems: list) -> int:
    return len(indices_of_true(elems))


def vector_mean(vectors: Sequence[list], summer_class=KahanSummer) -> list:
    n_vectors = len(vectors)
    vect_len = len(vectors[0])
    summers = [summer_class.create() for _ in range(vect_len)]
    for v in vectors:
        for i in range(vect_len):
            summers[i].add(v[i])
    return [summers[i].get_sum()/n_vectors for i in range(vect_len)]


def product_of_sequence(s: Sequence):
    return umath.multiply.reduce(s)


def top_k_positions(elems: list, k: int):
    """Returns indices of higher k elements, ordered from higher. Uses stable sorting."""
    a = np.array(elems)
    return np.argsort(-a, kind='stable')[:k]


def cartesian_product(lists):
    """Takes in input a list of lists."""
    return itertools.product(*lists)
