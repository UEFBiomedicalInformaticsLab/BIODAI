from abc import ABC, abstractmethod
from collections.abc import Sequence

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.solutions_from_files import solutions_from_files


class HofFoldPropertyComputer(ABC):

    @abstractmethod
    def compute(self, hof_path: str, fold: int) -> float:
        raise NotImplementedError()

    @abstractmethod
    def compute_all(self, hof_path: str) -> Sequence[float]:
        """Returns a measure for each fold."""
        raise NotImplementedError()


class HofFoldPropertyComputerFromFoldMeasure(HofFoldPropertyComputer):
    __measure: RunFoldMeasure

    def __init__(self, measure: RunFoldMeasure):
        self.__measure = measure

    def compute(self, hof_path: str, fold: int) -> float:
        solutions = solutions_from_files(hof_dir=hof_path)
        return self.__measure.compute_fold_measure(solutions=solutions[fold])

    def compute_all(self, hof_path: str) -> Sequence[float]:
        """Returns a measure for each fold."""
        solutions = solutions_from_files(hof_dir=hof_path)
        return self.__measure.compute_measures(solutions=solutions)
