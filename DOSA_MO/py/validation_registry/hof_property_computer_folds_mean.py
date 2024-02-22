from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.solutions_from_files import solutions_from_files

from util.math.summer import KahanSummer
from validation_registry.hof_property_computer import HofPropertyComputer, HofPropertyComputerWithFolds
from validation_registry.registry_property import smart_extract


class HofPropertyComputerFoldsMeanFromRegistry(HofPropertyComputerWithFolds):
    __folds_property_name: str
    __folds_property_computer: HofPropertyComputer

    def __init__(self, folds_property_name: str, folds_property_computer: HofPropertyComputer):
        self.__folds_property_name = folds_property_name
        self.__folds_property_computer = folds_property_computer

    def compute(self, hof_path: str) -> float:
        fold_values = smart_extract(
            prop_name=self.__folds_property_name, hof_path=hof_path, computer=self.__folds_property_computer)
        return KahanSummer.mean(fold_values)

    def compute_fold(self, hof_path: str, fold: int) -> float:
        """We do not try to extract value from registry since the compute method is expected to compute from scratch."""
        return self.__folds_property_computer.compute_fold(hof_path=hof_path, fold=fold)


class HofPropertyComputerFoldsMeanFromMeasure(HofPropertyComputerWithFolds):
    __measure: RunFoldMeasure

    def __init__(self, measure: RunFoldMeasure):
        self.__measure = measure

    def compute(self, hof_path: str) -> float:
        solutions = solutions_from_files(hof_dir=hof_path)
        folds_res = self.__measure.compute_measures(solutions=solutions)
        return KahanSummer.mean(folds_res)

    def compute_fold(self, hof_path: str, fold: int) -> float:
        solutions = solutions_from_files(hof_dir=hof_path)
        return self.__measure.compute_fold_measure(solutions=solutions[fold])
