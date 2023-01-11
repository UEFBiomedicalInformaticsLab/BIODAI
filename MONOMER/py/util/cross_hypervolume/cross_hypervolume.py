from itertools import compress
from typing import Sequence

from util.hyperbox.hyperbox import Hyperbox0B, Interval, ConcreteInterval
from util.list_math import list_multiply, cartesian_product
from util.summer import KahanSummer


def values_on_d(hyperboxes: [Hyperbox0B], d: int) -> [float]:
    return [h.intervals()[d].b() for h in hyperboxes]


def d_max_values(hyperboxes: [Hyperbox0B], d: int) -> [float]:
    """Does not return duplicates."""
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


def dimension_contribution(
        train_hyperboxes: [Hyperbox0B], volume_ratios: [float], d: int) -> float:
    n_hyperboxes = len(train_hyperboxes)
    vals_on_d = values_on_d(hyperboxes=train_hyperboxes, d=d)
    effectiveness = list_multiply(vals_on_d, volume_ratios)
    projections = [h.project(d) for h in train_hyperboxes]
    n_dimensions = train_hyperboxes[0].n_dimensions()
    intervals_lists = []
    for i in range(n_dimensions):
        if i is not d:
            intervals_lists.append(d_intervals(hyperboxes=train_hyperboxes, d=i))
    interval_combinations = cartesian_product(intervals_lists)
    res_summer = KahanSummer()
    for intervals in interval_combinations:
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        best_on_d = -1.0
        eff = [0.0]
        for h_index in range(n_hyperboxes):
            d_val = vals_on_d[h_index]
            if d_val >= best_on_d:
                if projections[h_index].contains_point(midpoint):
                    if d_val > best_on_d:
                        best_on_d = d_val
                        eff = [effectiveness[h_index]]
                    else:  # d_val == best_on_d
                        eff.append(effectiveness[h_index])
        # if len(eff) > 1:
        #     print("Solutions non-dominated with same performance. Effectiveness: " + str(eff) + "\n" +
        #           "Intervals: " + sequence_to_string(intervals))
        res_summer.add(i_volume*KahanSummer.mean(eff))
    return res_summer.get_sum()


def cross_hypervolume_on_front(train_hyperboxes: [Hyperbox0B], test_hyperboxes: [Hyperbox0B]) -> float:
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

    volume_ratios = []
    for test_h, train_h in zip(test_hyperboxes, train_hyperboxes):
        train_h_volume = train_h.volume()
        if train_h_volume == 0.0:  # No contribution is possible if train volume is 0
            volume_ratios.append(0.0)  # To avoid NaN
        else:
            volume_ratios.append(test_h.volume() / train_h_volume)
    dimension_contributions = [dimension_contribution(train_hyperboxes, volume_ratios, d) for d in range(n_dimensions)]
    # print("Dimension contributions: " + str(dimension_contributions))
    return KahanSummer.mean(dimension_contributions)


def non_dominated_hyperboxes(
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
    In zero dimensional space hypervolume defaults to zero. May contain dominated boxes."""
    front_train_hb, front_test_hb = non_dominated_hyperboxes(
        train_hyperboxes=train_hyperboxes, test_hyperboxes=test_hyperboxes)
    return cross_hypervolume_on_front(train_hyperboxes=front_train_hb, test_hyperboxes=front_test_hb)


def hypervolume(hyperboxes: Sequence[Hyperbox0B]) -> float:
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
    interval_combinations = cartesian_product(intervals_lists)
    res_summer = KahanSummer()
    for intervals in interval_combinations:
        midpoint = []
        i_volume = 1.0
        for i in intervals:
            i_volume = i_volume * i.length()
            midpoint.append(i.mid_pos())
        best_on_d = 0.0  # Must be 0 because it can happen that no projection contains the point.
        for h_index in range(n_hyperboxes):
            d_val = vals_on_d[h_index]
            if d_val > best_on_d:
                if projections[h_index].contains_point(midpoint):
                    best_on_d = d_val
        res_summer.add(i_volume * best_on_d)
    return res_summer.get_sum()
