from abc import ABC, abstractmethod

import numpy as np
import sklearn

from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from folds_creator.mo_class_assigner.mo_class_assigner import MOClassAssigner
from input_data.input_data import InputData
from util.printer.printer import Printer, NullPrinter


class InputDataKFoldsCreator(InputDataFoldsCreator, ABC):
    __n_folds: int
    __n_repeats: int

    def __init__(self, n_folds, n_repeats: int = 1):
        self.__n_folds = n_folds
        self.__n_repeats = n_repeats

    def n_folds(self) -> int:
        return self.__n_folds

    def n_repeats(self) -> int:
        return self.__n_repeats

    def create_folds_from_input_data(self, input_data: InputData, seed: int = 365, printer: Printer = NullPrinter()
                                     ) -> list[tuple]:
        x = np.zeros(input_data.n_samples())
        strata = self._class_assigner().assign_classes(input_data, printer=printer, min_stratum_size=self.n_folds())
        res = []
        skf = sklearn.model_selection.RepeatedStratifiedKFold(
            n_splits=self.n_folds(), n_repeats=self.n_repeats(), random_state=seed)
        for train_index, test_index in skf.split(X=x, y=strata.ids()):
            res.append((train_index, test_index))
        return res

    @abstractmethod
    def _class_assigner(self) -> MOClassAssigner:
        raise NotImplementedError()
