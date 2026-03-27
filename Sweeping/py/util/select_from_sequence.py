import itertools
from typing import Sequence, Iterable

import numpy as np

from model.sv_model import PredictProbaResult
from util.sequence_utils import true_positions_sorted_set


def select_by_indices(data: Sequence, indices: Iterable[int]) -> Sequence:
    """If the sequence is a DataFrame, the indexing will be by row names. Preserves order.
    Optimized for the case of data being a ndarray. If the sequence is a PredictProbaResult,
    the result is still a PredictProbaResult."""
    try:
        if isinstance(data, np.ndarray):
            return data[indices]
        elif isinstance(data, PredictProbaResult):
            return data.select_by_indices(indices=indices)
        else:
            return [data[i] for i in indices]
    except KeyError as e:
        raise KeyError("KeyError when accessing an element of " + str(data) + "\n" + str(e))


def select_by_mask(data: Sequence, mask: Sequence[bool]) -> Sequence:
    """If the sequence is a DataFrame, the indexing will be by row names.
    Optimized for ListLike masks."""
    if not len(data) == len(mask):
        raise ValueError("Different lengths.\n" +
                         "Data length: " + str(len(data)) + "\n" +
                         "Mask length: " + str(len(mask)) + "\n")
    return select_by_indices(data=data, indices=true_positions_sorted_set(s=mask))


def filter_by_booleans(data: Sequence, selectors: Sequence[bool]) -> list:
    if len(data) != len(selectors):
        raise ValueError("data and selectors must have the same length")
    return list(itertools.compress(data, selectors))


def first_elements(lst: Sequence, num_elems: int = 5) -> Sequence:
    """Returns the whole sequence if it contains less than num_elems."""
    return lst[:num_elems]


def last_elements(lst: Sequence, num_elems: int = 5) -> Sequence:
    """Returns the whole sequence if it contains less than num_elems."""
    return lst[-num_elems:]
