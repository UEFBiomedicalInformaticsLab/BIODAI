from collections.abc import Sequence

from util.math.list_math import mean_all_vs_others, list_tot_abs_difference
from util.utils import same_len


def stability_of_distributions(distributions: Sequence[Sequence[float]]):
    """Distributions are represented by sequences of floats, where the sum of each list must be 1. Each sequence
    must have the same length.
    TODO Can be optimized by computing the mean of all the distributions."""
    if not same_len(distributions):
        raise ValueError()
    if len(distributions) < 2:
        stability = 1.0
    else:
        stability = \
            1.0 - (mean_all_vs_others(elems=distributions, measure_function=list_tot_abs_difference) / 2.0)
    return stability
