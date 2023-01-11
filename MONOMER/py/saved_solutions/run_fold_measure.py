from abc import abstractmethod
from collections.abc import Sequence

from saved_solutions.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution
from util.summer import KahanSummer


class RunFoldMeasure(RunMeasure):

    @abstractmethod
    def compute_fold_measure(self, solutions: Sequence[SavedSolution]) -> float:
        """Take in input the solutions for a specific fold.
        Returns a measure for that fold."""
        raise NotImplementedError()

    def compute_measures(self, solutions: Sequence[Sequence[SavedSolution]]) -> Sequence[float]:
        """Take in input a sequence for each fold. The inner sequences are the solutions for each specific fold.
        Returns a measure for each fold."""
        return [self.compute_fold_measure(fold_solutions) for fold_solutions in solutions]

    def compute_measure(self, solutions: Sequence[Sequence[SavedSolution]]) -> float:
        measures = self.compute_measures(solutions)
        if len(measures) == 0:
            return 0.0
        else:
            return KahanSummer.mean(measures)
