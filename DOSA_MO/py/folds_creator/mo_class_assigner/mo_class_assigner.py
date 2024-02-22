from abc import ABC, abstractmethod

from folds_creator.mo_class_assigner.strata import Strata
from input_data.input_data import InputData
from util.printer.printer import Printer, NullPrinter


class MOClassAssigner(ABC):

    @abstractmethod
    def assign_classes(self, data: InputData, printer: Printer = NullPrinter(), min_stratum_size: int = 1) -> Strata:
        raise NotImplementedError()
