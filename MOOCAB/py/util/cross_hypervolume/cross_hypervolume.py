from itertools import compress
from typing import Sequence, Optional

from util.cross_hypervolume.hv_utils import DEFAULT_HV_PROCS, procs_to_use, intervals_lists_for_workers
from util.hyperbox.hyperbox import Hyperbox0B, Interval, ConcreteInterval
from util.math.list_math import list_multiply, cartesian_product, cartesian_product_size
from util.math.summer import KahanSummer
from concurrent.futures import ProcessPoolExecutor


def values_on_d(hyperboxes: [Hyperbox0B], d: int) -> list[float]:
    return [h.intervals()[d].b() for h in hyperboxes]


def d_max_values(hyperboxes: [Hyperbox0B], d: int) -> [float]:
    """Does not return duplicates. The maximum values of all the hyperboxes in the d direction."""
    return sorted(set(values_on_d(hyperboxes=hyperboxes, d=d)))


def d_midpositions(hyperboxes: [Hyperbox0B], d: int) -> [float]:
    d_max_vals = d_max_values(hyperboxes=hyperboxes, d=d)
    res = []
    lower = 0.0
    for upper in d_max_vals:
        res.append((lower+upper)/2.0)
        lower = upper
    return res


def d_intervals(hyperboxes: [Hyperbox0B], d: int) -> [Interval]:
    d_max_vals = d_max_values(hyperboxes=hyperboxes, d=d)
    res = []
    lower = 0.0
    for upper in d_max_vals:
        res.append(ConcreteInterval(lower, upper))
        lower = upper
    return res


def cross_interval_contribution(
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        effectiveness: Sequence[float],
        midpoint: Sequence[float],
        i_volume: float
        ) -> float:
    best_on_d = -1.0
    eff = [0.0]  # To avoid computing the mean of an empty list.
    for h_index in range(len(vals_on_d)):
        d_val = vals_on_d[h_index]
        if d_val >= best_on_d:
            if projections[h_index].contains_point(midpoint):
                if d_val > best_on_d:
                    best_on_d = d_val
                    eff = [effectiveness[h_index]]
                else:  # d_val == best_on_d
                    eff.append(effectiveness[h_index])  # Solutions non-dominated with same performance.
    return i_volume * KahanSummer.mean(eff)


def cross_evaluate_intervals_sequential(
        intervals_lists: Sequence[Sequence],
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        effectiveness: Sequence[float]) -> float:
    interval_combinations = cartesian_product(intervals_lists)
    res_summer = KahanSummer()
    for intervals in interval_combinations:  # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        res_summer.add(
                cross_interval_contribution(
                    vals_on_d=vals_on_d,
                    projections=projections,
                    effectiveness=effectiveness,
                    i_volume=i_volume,
                    midpoint=midpoint))
    return res_summer.get_sum()


class CrossEvaluateParallelInput:
    intervals_lists: Sequence[tuple]
    vals_on_d: Sequence[float]
    projections: list[Hyperbox0B]
    effectiveness: Sequence[float]


def cross_parallel_function(par_input: CrossEvaluateParallelInput) -> float:
    vals_on_d = par_input.vals_on_d
    projections = par_input.projections
    effectiveness = par_input.effectiveness
    res_summer = KahanSummer()
    for intervals in cartesian_product(par_input.intervals_lists):
        # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for j in intervals:
            i_volume = i_volume * j.length()
            midpoint.append(j.mid_pos())
        res_summer.add(
            cross_interval_contribution(
                vals_on_d=vals_on_d,
                projections=projections,
                effectiveness=effectiveness,
                i_volume=i_volume,
                midpoint=midpoint))
    return res_summer.get_sum()


def cross_evaluate_intervals_parallel(
        intervals_lists: Sequence[Sequence],
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        effectiveness: Sequence[float],
        n_proc: int) -> float:
    parallel_inputs = []
    for intervals_lists in intervals_lists_for_workers(lists=intervals_lists):
        pi = CrossEvaluateParallelInput()
        pi.vals_on_d = vals_on_d
        pi.projections = projections
        pi.effectiveness = effectiveness
        pi.intervals_lists = intervals_lists
        parallel_inputs.append(pi)
    with (ProcessPoolExecutor(max_workers=n_proc) as workers_pool):
        workers_res = workers_pool.map(
            cross_parallel_function, parallel_inputs, chunksize=1)
        try:
            list_res = [i for i in workers_res]
        except RecursionError as e:
            message = "Error while reading worker results.\n" + "Original error: " + str(e) + "\n"
            print(message)
            raise RecursionError(message)
    return KahanSummer().sum(list_res)


def cross_dimension_contribution(
        train_hyperboxes: Sequence[Hyperbox0B], volume_ratios: Sequence[float], d: int,
        n_proc: Optional[int] = DEFAULT_HV_PROCS) -> float:
    vals_on_d = values_on_d(hyperboxes=train_hyperboxes, d=d)
    n_dimensions = train_hyperboxes[0].n_dimensions()
    intervals_lists = []
    for i in range(n_dimensions):
        if i is not d:
            intervals_lists.append(d_intervals(hyperboxes=train_hyperboxes, d=i))
    projections = [h.project(d) for h in train_hyperboxes]
    effectiveness = list_multiply(vals_on_d, volume_ratios)
    n_proc = procs_to_use(
        n_proc=n_proc, n_intervals=cartesian_product_size(intervals_lists), n_solutions=len(train_hyperboxes))
    if n_proc <= 1:
        return cross_evaluate_intervals_sequential(
            intervals_lists=intervals_lists,
            vals_on_d=vals_on_d,
            projections=projections,
            effectiveness=effectiveness
        )
    else:
        return cross_evaluate_intervals_parallel(
            intervals_lists=intervals_lists,
            vals_on_d=vals_on_d,
            projections=projections,
            effectiveness=effectiveness,
            n_proc=n_proc)


def cross_hypervolume_on_front(
        train_hyperboxes: [Hyperbox0B], test_hyperboxes: [Hyperbox0B], verbose: bool = True) -> float:
    """The two list of hyperboxes must have the same length and the i-th element of both list must refer to the
    same solution, with performance on the train and on the test datasets.
    In zero dimensional space hypervolume defaults to zero.
    Assumes there are no dominated train boxes."""

    train_len = len(train_hyperboxes)
    test_len = len(test_hyperboxes)
    if train_len != test_len:
        raise ValueError("There must be the same number of train and test hyperboxes.")
    if train_len == 0:
        return 0.0
    n_dimensions = train_hyperboxes[0].n_dimensions()
    if n_dimensions == 0:
        return 0.0

    for h in train_hyperboxes:
        if h.n_dimensions() != n_dimensions:
            raise ValueError(
                "All hyperboxes must have the same number of dimensions.\n" +
                "Dimensions of first train hyperbox: " + str(n_dimensions) + "\n" +
                "Dimensions of other train hyperbox: " + str(h.n_dimensions()) + "\n")
    for h in test_hyperboxes:
        if h.n_dimensions() != n_dimensions:
            raise ValueError(
                "All hyperboxes must have the same number of dimensions.\n" +
                "Dimensions of first train hyperbox: " + str(n_dimensions) + "\n" +
                "Dimensions of a test hyperbox: " + str(h.n_dimensions()) + "\n")

    if verbose:
        print(
            "Computing cross hypervolume on " + str(train_len) + " solutions with " + str(n_dimensions) + " objectives")

    volume_ratios = []
    for test_h, train_h in zip(test_hyperboxes, train_hyperboxes):
        train_h_volume = train_h.volume()
        if train_h_volume == 0.0:  # No contribution is possible if train volume is 0
            volume_ratios.append(0.0)  # To avoid NaN
        else:
            volume_ratios.append(test_h.volume() / train_h_volume)
    dimension_contributions = [
        cross_dimension_contribution(train_hyperboxes, volume_ratios, d) for d in range(n_dimensions)]
    if verbose:
        print("Dimension contributions: " + str(dimension_contributions))
    return KahanSummer.mean(dimension_contributions)


def non_dominated_indices(
        hyperboxes: Sequence[Hyperbox0B]) -> Sequence[int]:
    res = []
    for i, h in enumerate(hyperboxes):
        if not h.is_dominated(hyperboxes):
            res.append(i)
    return res


def non_dominated_hyperboxes(
        hyperboxes: Sequence[Hyperbox0B]) -> Sequence[Hyperbox0B]:
    res = []
    for h in hyperboxes:
        if not h.is_dominated(hyperboxes):
            res.append(h)
    return res


def cross_non_dominated_hyperboxes(
        train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B]) -> ([Hyperbox0B], [Hyperbox0B]):
    n_boxes = len(train_hyperboxes)
    mask = [True]*n_boxes
    for i in range(n_boxes):
        temp_train = train_hyperboxes[i]
        if temp_train.is_dominated(train_hyperboxes):
            mask[i] = False
    return list(compress(train_hyperboxes, mask)), list(compress(test_hyperboxes, mask))


def cross_hypervolume(train_hyperboxes: Sequence[Hyperbox0B], test_hyperboxes: Sequence[Hyperbox0B]) -> float:
    """The two list of hyperboxes must have the same length and the i-th element of both list must refer to the
    same solution, with performance on the train and on the test datasets.
    In zero dimensional space hypervolume defaults to zero. May contain dominated boxes.
    TODO Polymorphic function generalizing Pareto delta and cross HV."""
    front_train_hb, front_test_hb = cross_non_dominated_hyperboxes(
        train_hyperboxes=train_hyperboxes, test_hyperboxes=test_hyperboxes)
    return cross_hypervolume_on_front(train_hyperboxes=front_train_hb, test_hyperboxes=front_test_hb)


def hv_interval_contribution(
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        midpoint: Sequence[float],
        i_volume: float
        ) -> float:
    best_on_d = 0.0  # Must be 0 because it can happen that no projection contains the point.
    for h_index in range(len(vals_on_d)):
        d_val = vals_on_d[h_index]
        if d_val > best_on_d:
            if projections[h_index].contains_point(midpoint):
                best_on_d = d_val
    return i_volume * best_on_d


def hv_evaluate_intervals_sequential(
        intervals_lists: Sequence[Sequence],
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B]) -> float:
    res_summer = KahanSummer()
    interval_combinations = cartesian_product(lists=intervals_lists)
    for intervals in interval_combinations:  # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        res_summer.add(
                hv_interval_contribution(
                    vals_on_d=vals_on_d,
                    projections=projections,
                    i_volume=i_volume,
                    midpoint=midpoint))
    return res_summer.get_sum()


class HvEvaluateParallelInput:
    intervals_lists: Sequence[tuple]
    vals_on_d: Sequence[float]
    projections: list[Hyperbox0B]


def hv_parallel_function(par_input: HvEvaluateParallelInput) -> float:
    vals_on_d = par_input.vals_on_d
    projections = par_input.projections
    res_summer = KahanSummer()
    for intervals in cartesian_product(par_input.intervals_lists):
        # intervals is one interval for each dimension except d.
        midpoint = []
        i_volume = 1.0
        for j in intervals:
            i_volume = i_volume * j.length()
            midpoint.append(j.mid_pos())
        res_summer.add(
            hv_interval_contribution(
                vals_on_d=vals_on_d,
                projections=projections,
                i_volume=i_volume,
                midpoint=midpoint))
    return res_summer.get_sum()


def hv_evaluate_intervals_parallel(
        intervals_lists: Sequence[Sequence],
        vals_on_d: Sequence[float],
        projections: list[Hyperbox0B],
        n_proc: int) -> float:
    parallel_inputs = []
    for intervals_lists in intervals_lists_for_workers(lists=intervals_lists):
        pi = HvEvaluateParallelInput()
        pi.vals_on_d = vals_on_d
        pi.projections = projections
        pi.intervals_lists = intervals_lists
        parallel_inputs.append(pi)
    with ProcessPoolExecutor(max_workers=n_proc) as workers_pool:
        workers_res = workers_pool.map(
            hv_parallel_function, parallel_inputs, chunksize=1)
        res = KahanSummer()
        try:
            for i in workers_res:
                res.add(i)
        except RecursionError as e:
            raise RecursionError("Original error: " + str(e) + "\n" +
                                 "while appending:\n")
    return res.get_sum()


def hypervolume(hyperboxes: Sequence[Hyperbox0B],
                n_proc: Optional[int] = DEFAULT_HV_PROCS,
                verbose: bool = True) -> float:
    """In zero dimensional space hypervolume defaults to zero."""
    n_hyperboxes = len(hyperboxes)
    if n_hyperboxes == 0:
        return 0.0
    n_dimensions = hyperboxes[0].n_dimensions()
    if n_dimensions == 0:
        return 0.0
    for h in hyperboxes:
        if h.n_dimensions() != n_dimensions:
            raise ValueError("All hyperboxes must have the same number of dimensions.")
    vals_on_d = values_on_d(hyperboxes=hyperboxes, d=0)
    projections = [h.project(0) for h in hyperboxes]
    intervals_lists = []
    for i in range(1, n_dimensions):
        intervals_lists.append(d_intervals(hyperboxes=hyperboxes, d=i))
    n_proc = procs_to_use(
        n_proc=n_proc, n_intervals=cartesian_product_size(intervals_lists), n_solutions=n_hyperboxes)
    if verbose:
        print("Expected comparisons: " + str(cartesian_product_size(intervals_lists)*n_hyperboxes))
        print("Using procs: " + str(n_proc))
    if n_proc <= 1:
        return hv_evaluate_intervals_sequential(
            intervals_lists=intervals_lists,
            vals_on_d=vals_on_d,
            projections=projections
        )
    else:
        return hv_evaluate_intervals_parallel(
            intervals_lists=intervals_lists,
            vals_on_d=vals_on_d,
            projections=projections,
            n_proc=n_proc)
