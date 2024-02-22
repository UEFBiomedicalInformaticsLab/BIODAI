from collections.abc import Sequence

from util.cross_hypervolume.cross_hypervolume import non_dominated_indices, values_on_d, d_intervals
from util.hyperbox.hb_utils import check_dimensions_all
from util.hyperbox.hyperbox import Hyperbox0B
from util.math.list_math import list_subtract, list_abs, cartesian_product
from util.math.summer import KahanSummer
from util.preconditions import check_same_len
from util.sequence_utils import select_by_indices

PARETO_DELTA_NAME = "Pareto delta"
PARETO_DELTA_NICK = "pareto_delta"


def dimension_pareto_delta(
        train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B], d: int) -> float:
    """If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    d is the index of the dimension with respect to which the Pareto delta has to be computed.
    It is assumed that there are no dominated train hyperboxes and the train and test hyperboxes in the same position
    are related to the same solution."""
    n_hyperboxes = len(train_hyperboxes)
    train_vals_on_d = values_on_d(hyperboxes=train_hyperboxes, d=d)
    test_vals_on_d = values_on_d(hyperboxes=test_hyperboxes, d=d)
    deltas = list_abs(list_subtract(list_a=train_vals_on_d, list_b=test_vals_on_d))
    projections = [h.project(d) for h in train_hyperboxes]
    n_dimensions = train_hyperboxes[0].n_dimensions()
    intervals_lists = []
    for i in range(n_dimensions):
        if i is not d:
            intervals_lists.append(d_intervals(hyperboxes=train_hyperboxes, d=i))
    interval_combinations = cartesian_product(intervals_lists)
    res_summer = KahanSummer()
    for intervals in interval_combinations:  # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        best_on_d = -1.0
        best_h_indices = []
        for h_index in range(n_hyperboxes):
            d_val = train_vals_on_d[h_index]
            if d_val >= best_on_d:
                if projections[h_index].contains_point(midpoint):
                    if d_val > best_on_d:
                        best_on_d = d_val
                        best_h_indices = [h_index]
                    else:  # d_val == best_on_d
                        best_h_indices.append(h_index)  # Solutions non-dominated with same performance.
        n_tied = len(best_h_indices)
        if n_tied > 0:
            volume_quota = i_volume/(float(n_tied))
            for i in best_h_indices:
                res_summer.add(volume_quota * deltas[i])
    return res_summer.get_sum()


def pareto_delta_on_front(
        train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B], verbose: bool = True) -> float:
    """
    Train and test hyperboxes at the same position must be related to the same solution.
    Assumes there are no dominated boxes.
    If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    """

    check_same_len(train_hyperboxes, test_hyperboxes)
    n_boxes = len(train_hyperboxes)
    if n_boxes == 0:
        return 0.0
    n_dimensions = train_hyperboxes[0].n_dimensions()
    check_dimensions_all(hyperboxes=train_hyperboxes, expected_dimensions=n_dimensions)
    check_dimensions_all(hyperboxes=test_hyperboxes, expected_dimensions=n_dimensions)
    if n_dimensions == 0:
        return 0.0

    if verbose:
        print(
            "Computing Pareto delta on " + str(n_boxes) + " solutions with "
            + str(n_dimensions) + " objectives")

    all_dimensions_pareto_deltas = [
        dimension_pareto_delta(
            train_hyperboxes=train_hyperboxes, test_hyperboxes=test_hyperboxes, d=d) for d in range(n_dimensions)]

    return KahanSummer.mean(all_dimensions_pareto_deltas)


def pareto_delta(train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B]) -> float:
    """
    Train and test hyperboxes at the same position must be related to the same solution.
    If two or more solutions have the same fitnesses, the derivative would not be defined.
    Instead, we compute the derivative as if there was just one solution, then divide by the number of tied solutions.
    """
    front_indices = non_dominated_indices(hyperboxes=train_hyperboxes)
    return pareto_delta_on_front(
        train_hyperboxes=select_by_indices(data=train_hyperboxes, indices=front_indices),
        test_hyperboxes=select_by_indices(data=test_hyperboxes, indices=front_indices))
