from collections.abc import Sequence

from util.cross_hypervolume.cross_hypervolume import non_dominated_hyperboxes, values_on_d, d_intervals
from util.distribution.distribution import ConcreteDistribution
from util.hyperbox.hyperbox import Hyperbox0B
from util.math.list_math import cartesian_product
from util.math.summer import KahanSummer
from util.sequence_utils import transpose


def dimension_derivatives(
        hyperboxes: Sequence[Hyperbox0B], d: int) -> list[float]:
    """If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    This is a good compromise when, e.g., using the derivative to assign weights to errors in fitness assessment.
    d is the index of the dimension with respect to which the derivatives have to be computed.
    TODO Possible target for parallelization."""
    n_hyperboxes = len(hyperboxes)
    vals_on_d = values_on_d(hyperboxes=hyperboxes, d=d)
    projections = [h.project(d) for h in hyperboxes]
    n_dimensions = hyperboxes[0].n_dimensions()
    intervals_lists = []
    for i in range(n_dimensions):
        if i is not d:
            intervals_lists.append(d_intervals(hyperboxes=hyperboxes, d=i))
    interval_combinations = cartesian_product(intervals_lists)
    res_summers = [KahanSummer() for _ in range(n_hyperboxes)]
    for intervals in interval_combinations:  # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        best_on_d = -1.0
        best_h_indices = []
        for h_index in range(n_hyperboxes):
            d_val = vals_on_d[h_index]
            if d_val >= best_on_d:
                if projections[h_index].contains_point(midpoint):
                    if d_val > best_on_d:
                        best_on_d = d_val
                        best_h_indices = [h_index]
                    else:  # d_val == best_on_d
                        best_h_indices.append(h_index)  # Solutions non-dominated with same performance.
        n_tied = len(best_h_indices)
        if n_tied > 0:
            to_add = i_volume/(float(n_tied))
            for i in best_h_indices:
                res_summers[i].add(to_add)
    return [r.get_sum() for r in res_summers]


def hypervolume_derivatives_on_front(hyperboxes: Sequence[Hyperbox0B], verbose: bool = True) -> list[list[float]]:
    """Assumes there are no dominated boxes.
    Returns a list for solutions, containing a list with the derivative with respect to each objective.
    If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    This is a good compromise when, e.g., using the derivative to assign weights to errors in fitness assessment."""

    n_boxes = len(hyperboxes)
    if n_boxes == 0:
        return []
    n_dimensions = hyperboxes[0].n_dimensions()
    if n_dimensions == 0:
        return [[] for _ in hyperboxes]

    for h in hyperboxes:
        if h.n_dimensions() != n_dimensions:
            raise ValueError(
                "All hyperboxes must have the same number of dimensions.\n" +
                "Dimensions of first hyperbox: " + str(n_dimensions) + "\n" +
                "Dimensions of other hyperbox: " + str(h.n_dimensions()) + "\n")

    if verbose:
        print(
            "Computing hypervolume derivatives on " + str(n_boxes) + " solutions with "
            + str(n_dimensions) + " objectives")

    all_dimensions_derivatives = [dimension_derivatives(hyperboxes, d) for d in range(n_dimensions)]

    return transpose(all_dimensions_derivatives)


def hypervolume_derivatives(hyperboxes: Sequence[Hyperbox0B]) -> list[list[float]]:
    """May contain dominated boxes.
    Returns a list for solutions, containing a list with the derivative with respect to each objective.
    If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    This is a good compromise when, e.g., using the derivative to assign weights to errors in fitness assessment."""
    front_hb = non_dominated_hyperboxes(hyperboxes=hyperboxes)
    return hypervolume_derivatives_on_front(hyperboxes=front_hb)


def normalized_hypervolume_derivatives(hyperboxes: Sequence[Hyperbox0B]) -> list[list[float]]:
    """May contain dominated boxes.
    Returns a list for solutions, containing a list with the derivative with respect to each objective.
    For each objective, the derivatives are normalized to sum to 1.
    If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    This is a good compromise when, e.g., using the derivative to assign weights to errors in fitness assessment."""
    front_hb = non_dominated_hyperboxes(hyperboxes=hyperboxes)
    res_transposed = transpose(hypervolume_derivatives_on_front(hyperboxes=front_hb))
    return transpose([ConcreteDistribution(d) for d in res_transposed])
