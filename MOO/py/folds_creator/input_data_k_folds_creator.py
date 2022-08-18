import sklearn

from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from input_data.input_data import InputData


class InputDataKFoldsCreator(InputDataFoldsCreator):
    __n_folds: int

    def __init__(self, n_folds):
        self.__n_folds = n_folds

    def create_folds_from_input_data(self, input_data: InputData, seed: int = 365):
        x = input_data.collapsed_views()
        y = input_data.stratify_outcome_data()
        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=self.__n_folds, shuffle=True, random_state=seed)
        res = []
        for train_index, test_index in skf.split(X=x, y=y):
            # TODO It is not clear what is done by split when y has two or more columns and/or float values.
            res.append([train_index, test_index])
        return res

    def name(self) -> str:
        return str(self.__n_folds) + "-folds creator"

    def nick(self) -> str:
        return "k" + str(self.__n_folds)

    def __str__(self) -> str:
        return "input data " + str(self.__n_folds) + "-folds creator"
