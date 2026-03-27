from abc import abstractmethod

from folds_creator.folds_creator import FoldsCreator
from folds_creator.index_array import IndexArray
from input_data.input_data import InputData
from util.printer.printer import Printer, NullPrinter


class InputDataFoldsCreator(FoldsCreator):

    @abstractmethod
    def create_folds_from_input_data(self, input_data: InputData, seed: int = 365, printer: Printer = NullPrinter()
                                     ) -> list[tuple[IndexArray,IndexArray]]:
        # For each fold a sequence for train at position 0 and another for test at position 1.
        raise NotImplementedError()
