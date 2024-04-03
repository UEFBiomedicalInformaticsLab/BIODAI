from abc import abstractmethod
from collections.abc import Sequence

from saved_solutions.saved_solution import SavedSolution
from util.named import NickNamed


class RunMeasure(NickNamed):

    @abstractmethod
    def compute_measure(self, solutions: Sequence[Sequence[SavedSolution]]) -> float:
        """Take in input a sequence for each fold. The inner sequences are the solutions for each specific fold."""
        raise NotImplementedError()
