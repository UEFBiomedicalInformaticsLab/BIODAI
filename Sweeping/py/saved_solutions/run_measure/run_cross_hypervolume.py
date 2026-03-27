from typing import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.cross_hypervolume.cross_hypervolume import cross_hypervolume
from validation_registry.allowed_property_names import CROSS_HV_NAME, CROSS_HV_NICK


class RunCrossHypervolume(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        return cross_hypervolume(
            train_hyperboxes=[s.train_hyperbox() for s in solutions],
            test_hyperboxes=[s.test_hyperbox() for s in solutions])

    def name(self) -> str:
        return CROSS_HV_NAME

    def nick(self) -> str:
        return CROSS_HV_NICK
