from typing import Sequence

from saved_solutions.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.cross_hypervolume.cross_hypervolume import cross_hypervolume


class RunSoftCrossHypervolume(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        """Converts fitness in position 1 from leanness to soft-leanness"""
        return cross_hypervolume(
            train_hyperboxes=[s.train_soft_hyperbox() for s in solutions],
            test_hyperboxes=[s.test_soft_hyperbox() for s in solutions])

    def name(self) -> str:
        return "soft cross hypervolume"

    def nick(self) -> str:
        return "soft_cross_hypervol"
