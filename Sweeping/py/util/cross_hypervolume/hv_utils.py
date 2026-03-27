import multiprocessing
from typing import Optional, Sequence

DEFAULT_HV_PROCS = None

MAX_PROCS_DIVISOR = 1000000
# The higher, the fewer processors are used.
# After some short tests it seems that it has to be at least 500000.


def max_procs(n_intervals: int, n_solutions: int) -> int:
    return max(1, (n_intervals*n_solutions)//MAX_PROCS_DIVISOR)


def procs_to_use(n_intervals: int, n_solutions: int, n_proc: Optional[int]) -> int:
    cpu_count = multiprocessing.cpu_count()
    if n_proc is None:
        n_proc = cpu_count
    else:
        n_proc = min(n_proc, cpu_count)
    return min(n_proc, max_procs(n_intervals=n_intervals, n_solutions=n_solutions))


def intervals_lists_for_workers(lists: Sequence[Sequence]) -> Sequence[Sequence[Sequence]]:
    max_len = -1
    max_len_pos = -1
    for i, e in enumerate(lists):
        e_len = len(e)
        if e_len > max_len:
            max_len = e_len
            max_len_pos = i
    res = []
    for e in lists[max_len_pos]:
        current_list = []
        for i, li in enumerate(lists):
            if i == max_len_pos:
                current_list.append([e])
            else:
                current_list.append(li)
        res.append(current_list)
    return res
