import json
from collections.abc import Iterable

from sortedcontainers import SortedSet

from util.sequence_utils import str_in_lines


class Folds:
    """Supports also repeated CV. It is assumed that the union of all test sets gives all the samples."""
    __test_sets: list[SortedSet[int]]
    __all_samples: SortedSet[int]

    def __init__(self, test_sets: Iterable[Iterable[int]]):
        """The test sets are passed."""
        whole = SortedSet()
        self.__test_sets = []
        self.__n_samples = 0
        for input_test_set in test_sets:
            new_set = SortedSet(input_test_set)
            self.__test_sets.append(new_set)
            whole = whole.union(new_set)
        self.__all_samples = whole

    def train_indices(self, fold_number: int) -> SortedSet[int]:
        return self.__all_samples.difference(self.__test_sets[fold_number])

    def test_indices(self, fold_number: int) -> SortedSet[int]:
        return self.__test_sets[fold_number]

    def all_test_sets(self) -> list[SortedSet[int]]:
        return self.__test_sets

    def n_folds(self) -> int:
        return len(self.__test_sets)

    def __str__(self) -> str:
        return str_in_lines(self.__test_sets)


def save_folds(folds: Folds, file_path: str):
    raw_folds = folds.all_test_sets()
    raw_folds = [[int(x) for x in f] for f in raw_folds]
    with open(file_path, 'w') as fp:
        json.dump(raw_folds, fp)


def load_folds(file_path: str) -> Folds:
    with open(file_path) as f:
        raw_folds = json.load(f)
    return Folds(test_sets=raw_folds)
