from typing import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.cross_hypervolume.cross_hypervolume import hypervolume
from validation_registry.allowed_property_names import TEST_HV_NAME, TEST_HV_NICK


class RunTestHypervolume(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        return hypervolume([s.test_hyperbox() for s in solutions])

    def name(self) -> str:
        return TEST_HV_NAME

    def nick(self) -> str:
        return TEST_HV_NICK
