from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from util.cross_hypervolume.cross_hypervolume import non_dominated_indices, values_on_d, d_intervals
from util.cross_hypervolume.hv_utils import DEFAULT_HV_PROCS, procs_to_use, intervals_lists_for_workers
from util.hyperbox.hb_utils import check_dimensions_all
from util.hyperbox.hyperbox import Hyperbox0B
from util.math.list_math import list_subtract, list_abs, cartesian_product, cartesian_product_size
from util.math.summer import KahanSummer
from util.preconditions import check_same_len
from util.sequence_utils import select_by_indices
import deprecation

PARETO_DELTA_NAME = "Pareto delta"
PARETO_DELTA_NICK = "pareto_delta"


@deprecation.deprecated(details="Not used and maintained anymore.")
def dimension_pareto_delta_old(
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


def pareto_delta_interval_contribution(
        train_vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        deltas: Sequence[float],
        i_volume: float,
        midpoint: Sequence[float]
        ):
    best_on_d = -1.0
    best_deltas = [0.0]  # To avoid computing the mean of an empty list.
    for h_index in range(len(train_vals_on_d)):
        d_val = train_vals_on_d[h_index]
        if d_val >= best_on_d:
            if projections[h_index].contains_point(midpoint):
                if d_val > best_on_d:
                    best_on_d = d_val
                    best_deltas = [deltas[h_index]]
                else:  # d_val == best_on_d
                    best_deltas.append(deltas[h_index])  # Solutions non-dominated with same performance.
    return i_volume * KahanSummer.mean(best_deltas)


def pareto_delta_evaluate_intervals_sequential(
            intervals_lists: Sequence[Sequence],
            train_vals_on_d: Sequence[float],
            projections: list[Hyperbox0B],
            deltas: Sequence[float]):
    interval_combinations = cartesian_product(intervals_lists)
    res_summer = KahanSummer()
    for intervals in interval_combinations:  # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        res_summer.add(
            pareto_delta_interval_contribution(
                train_vals_on_d=train_vals_on_d,
                projections=projections,
                deltas=deltas,
                i_volume=i_volume,
                midpoint=midpoint))
    return res_summer.get_sum()


class ParetoDeltaEvaluateParallelInput:
    intervals_lists: Sequence[tuple]
    vals_on_d: Sequence[float]
    projections: list[Hyperbox0B]
    deltas: Sequence[float]


def pareto_delta_parallel_function(par_input: ParetoDeltaEvaluateParallelInput) -> float:
    vals_on_d = par_input.vals_on_d
    projections = par_input.projections
    deltas = par_input.deltas
    res_summer = KahanSummer()
    for intervals in cartesian_product(par_input.intervals_lists):
        # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for j in intervals:
            i_volume = i_volume * j.length()
            midpoint.append(j.mid_pos())
        res_summer.add(
            pareto_delta_interval_contribution(
                train_vals_on_d=vals_on_d,
                projections=projections,
                deltas=deltas,
                i_volume=i_volume,
                midpoint=midpoint))
    return res_summer.get_sum()


def pareto_delta_evaluate_intervals_parallel(
            intervals_lists: Sequence[Sequence],
            train_vals_on_d: Sequence[float],
            projections: list[Hyperbox0B],
            deltas: Sequence[float],
            n_proc: int):
    # print("Pareto delta with processors " + str(n_proc))
    parallel_inputs = []
    for intervals_lists in intervals_lists_for_workers(lists=intervals_lists):
        pi = ParetoDeltaEvaluateParallelInput()
        pi.vals_on_d = train_vals_on_d
        pi.projections = projections
        pi.deltas = deltas
        pi.intervals_lists = intervals_lists
        parallel_inputs.append(pi)
    with ProcessPoolExecutor(max_workers=n_proc) as workers_pool:
        workers_res = workers_pool.map(
            pareto_delta_parallel_function, parallel_inputs, chunksize=1)
        try:
            list_res = [i for i in workers_res]
        except RecursionError as e:
            message = "Error while reading worker results.\n" + "Original error: " + str(e) + "\n"
            print(message)
            raise RecursionError(message)
    return KahanSummer().sum(list_res)


def dimension_pareto_delta(
        train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B], d: int,
        n_proc: Optional[int] = DEFAULT_HV_PROCS) -> float:
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
    n_proc = procs_to_use(
        n_proc=n_proc, n_intervals=cartesian_product_size(intervals_lists), n_solutions=n_hyperboxes)
    if n_proc <= 1:
        return pareto_delta_evaluate_intervals_sequential(
            intervals_lists=intervals_lists,
            train_vals_on_d=train_vals_on_d,
            projections=projections,
            deltas=deltas)
    else:
        return pareto_delta_evaluate_intervals_parallel(
            intervals_lists=intervals_lists,
            train_vals_on_d=train_vals_on_d,
            projections=projections,
            deltas=deltas,
            n_proc=n_proc)


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
    TODO Polymorphic function generalizing Pareto delta and cross HV.
    """
    front_indices = non_dominated_indices(hyperboxes=train_hyperboxes)
    return pareto_delta_on_front(
        train_hyperboxes=select_by_indices(data=train_hyperboxes, indices=front_indices),
        test_hyperboxes=select_by_indices(data=test_hyperboxes, indices=front_indices))
