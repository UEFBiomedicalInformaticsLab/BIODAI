from abc import abstractmethod
from collections.abc import Sequence

from pandas import DataFrame

from folds_creator.folds_creator import FoldsCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from util.printer.printer import Printer, NullPrinter


class InputDataFoldsCreator(FoldsCreator):

    @abstractmethod
    def create_folds_from_input_data(self, input_data: InputData, seed: int = 365, printer: Printer = NullPrinter()
                                     ) -> Sequence[Sequence[Sequence[int]]]:
        # For each fold a sequence for train at position 0 and another for test at position 1.
        raise NotImplementedError()

    def create_folds_categorical(self, x: DataFrame, y: DataFrame, seed: int = 365):
        input_data = InputData(
            views={"x": x}, outcomes=[CategoricalOutcome(data=y, name="y")], nick="input")
        return self.create_folds_from_input_data(input_data=input_data, seed=seed)
