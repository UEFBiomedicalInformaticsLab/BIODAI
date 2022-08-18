from abc import abstractmethod

from folds_creator.folds_creator import FoldsCreator
from input_data.input_data import InputData


class InputDataFoldsCreator(FoldsCreator):

    @abstractmethod
    def create_folds_from_input_data(self, input_data: InputData, seed: int = 365):
        raise NotImplementedError()

    def create_folds(self, x, y, seed: int = 365):
        input_data = InputData(views={"x": x}, outcomes={"y": y}, nick="input")
        return self.create_folds_from_input_data(input_data=input_data)
