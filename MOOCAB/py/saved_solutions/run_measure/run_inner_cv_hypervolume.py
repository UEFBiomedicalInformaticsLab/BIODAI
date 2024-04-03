from typing import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.cross_hypervolume.cross_hypervolume import hypervolume
from validation_registry.allowed_property_names import INNER_CV_HV_NAME, INNER_CV_HV_NICK


class RunInnerCVHypervolume(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        hyperboxes = []
        for s in solutions:
            if s.has_train_fitnesses():
                hyperboxes.append(s.train_hyperbox())
        return hypervolume(hyperboxes=hyperboxes)

    def name(self) -> str:
        return INNER_CV_HV_NAME

    def nick(self) -> str:
        return INNER_CV_HV_NICK
