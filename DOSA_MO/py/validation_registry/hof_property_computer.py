from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.run_measure.run_measure import RunMeasure
from saved_solutions.solutions_from_files import solutions_from_files
from util.utils import IllegalStateError
from validation_registry.hof_fold_property_computer import HofFoldPropertyComputer, \
    HofFoldPropertyComputerFromFoldMeasure


class HofPropertyComputer(ABC):

    @abstractmethod
    def compute(self, hof_path: str) -> Any:
        raise NotImplementedError()

    def has_folds(self) -> bool:
        """True if the property has a value for each fold."""
        return False

    def compute_fold(self, hof_path: str, fold: int) -> float:
        raise IllegalStateError()


class HofPropertyComputerWithFolds(HofPropertyComputer, ABC):

    def has_folds(self) -> bool:
        return True

    @abstractmethod
    def compute_fold(self, hof_path: str, fold: int) -> float:
        raise NotImplementedError()


class HofPropertyComputerOneMeasure(HofPropertyComputer):
    __measure: RunMeasure

    def __init__(self, measure: RunMeasure):
        self.__measure = measure

    def compute(self, hof_path: str) -> float:
        solutions = solutions_from_files(hof_dir=hof_path)
        return self.__measure.compute_measure(solutions=solutions)

    def has_folds(self) -> bool:
        return False

    def compute_fold(self, hof_path: str, fold: int) -> float:
        raise IllegalStateError()


class HofPropertyComputerFromFoldMeasure(HofPropertyComputerWithFolds):
    __fold_computer: HofFoldPropertyComputer

    def __init__(self, measure: RunFoldMeasure):
        self.__fold_computer = HofFoldPropertyComputerFromFoldMeasure(measure=measure)

    def compute(self, hof_path: str) -> Sequence[float]:
        return self.__fold_computer.compute_all(hof_path=hof_path)

    def compute_fold(self, hof_path: str, fold: int) -> float:
        return self.__fold_computer.compute(hof_path=hof_path, fold=fold)
