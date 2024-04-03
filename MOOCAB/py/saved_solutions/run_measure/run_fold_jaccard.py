from collections.abc import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.math.summer import KahanSummer
from validation_registry.allowed_property_names import MEAN_JACCARD_NAME, MEAN_JACCARD_NICK


class RunFoldJaccard(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        features = []
        for s in solutions:
            if s.has_features():
                features.append(s.features())
        return compute_jaccard_from_lists(features)

    def name(self) -> str:
        return MEAN_JACCARD_NAME

    def nick(self) -> str:
        return MEAN_JACCARD_NICK


def jaccard_of_lists(list1, list2, zero_division=1.0) -> float:
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    if union == 0:
        return zero_division
    else:
        return float(intersection) / union


def compute_jaccard_from_lists(features: Sequence[Sequence[str]]) -> float:
    n_solutions = len(features)
    if n_solutions > 1:
        summation = KahanSummer()
        for j in range(n_solutions):
            for k in range(j + 1, n_solutions):
                summation.add(jaccard_of_lists(list1=features[j],
                                               list2=features[k],
                                               zero_division=1.0))
                # We consider to have perfect concordance when both individuals have no features.
        denominator = ((n_solutions * n_solutions) - n_solutions) / 2
        return summation.get_sum() / denominator
    else:
        return 1.0  # If there are zero or one individuals then the concordance of the features is total