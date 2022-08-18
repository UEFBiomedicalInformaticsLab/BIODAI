from util.hyperbox.hyperbox import Hyperbox0B, Interval, ConcreteInterval
from util.list_math import cartesian_product
from util.summer import KahanSummer


def values_on_d(hyperboxes: [Hyperbox0B], d: int) -> [float]:
    return [h.intervals()[d].b() for h in hyperboxes]


def d_max_values(hyperboxes: [Hyperbox0B], d: int) -> [float]:
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


def hypervolume(hyperboxes: [Hyperbox0B]) -> float:
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
