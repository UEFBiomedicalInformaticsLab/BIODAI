from typing import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.saved_solution import SavedSolution
from util.cross_hypervolume.pareto_delta import pareto_delta, PARETO_DELTA_NICK, PARETO_DELTA_NAME


class RunFoldParetoDelta(RunFoldMeasure):

    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        return pareto_delta(
            train_hyperboxes=[s.train_hyperbox() for s in solutions],
            test_hyperboxes=[s.test_hyperbox() for s in solutions])

    def name(self) -> str:
        return PARETO_DELTA_NAME

    def nick(self) -> str:
        return PARETO_DELTA_NICK
