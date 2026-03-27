from collections.abc import Sequence

from input_data.view_prefix import remove_view_prefix
from plots.runnable.best_solutions_printer import best_solutions_for_dataset
from plots.saved_hof import SavedHoF
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm


def best_genes_for_dataset(hofs: Sequence[SavedHoF], max_size: int = 100) -> dict[str, int]:
    """Solutions with zero or more than max_size features are excluded.
    Solutions with the same feature set are counted only once."""
    hofers = best_solutions_for_dataset(hofs=hofs)
    feature_sets = set()
    features = set()
    for h in hofers:
        if isinstance(h, SolutionFromAlgorithm):
            size = h.num_features()
            if 0 < size <= max_size:
                h_features = frozenset(h.solution().features())
                feature_sets.add(h_features)
                features.update(h_features)
    res = {}
    for f in features:
        count = 0
        for fs in feature_sets:
            if f in fs:
                count += 1
        res[f] = count
    return res


def best_genes_for_dataset_str(hofs: Sequence[SavedHoF], max_size: int = 100) -> str:
    best_genes = best_genes_for_dataset(hofs=hofs, max_size=max_size)
    entry_list = [(best_genes[g], (remove_view_prefix(g))[0]) for g in best_genes]
    entry_list.sort(key=lambda x: x[1])
    entry_list.sort(key=lambda x: x[0], reverse=True)
    res = ""
    for e in entry_list:
        res += str(e[1]) + ", " + str(e[0]) + "\n"
    return res
