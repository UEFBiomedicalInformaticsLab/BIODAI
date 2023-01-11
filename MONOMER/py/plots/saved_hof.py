import os
from typing import Optional, Sequence

from pandas import DataFrame

from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR
from plots.hof_utils import is_external_dir, test_df, external_df, to_test_dfs
from plots.performance_by_class import read_all_cms
from prediction_stats.confusion_matrix import ConfusionMatrix
from util.named import Named


class SavedHoF(Named):
    __path: str
    __name: str
    __main_algorithm_label: str

    def __init__(self, path: str, name: str, main_algorithm_label: str = ""):
        self.__path = path
        self.__name = name
        self.__main_algorithm_label = main_algorithm_label

    def path(self) -> str:
        return self.__path

    def name(self) -> str:
        return self.__name

    def main_algorithm_label(self) -> str:
        return self.__main_algorithm_label

    def to_df(self) -> Optional[DataFrame]:
        """If it is not possible to create a df returns None"""
        f = self.path()
        df = None
        if os.path.isdir(f):
            if is_external_dir(hof_dir=f):
                df = external_df(hof_dir=f)
            else:
                df = test_df(hof_dir=f)
            if df is None:
                print("Unable to create dataframe from directory " + str(f))
        else:
            print("path is not a directory: " + str(f))
        return df

    def to_dfs(self) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return to_test_dfs(hof_dir=self.path())

    def confusion_matrices(self) -> list[ConfusionMatrix]:
        """Fold files are read in alphabetical order. If passed path is not a directory returns an empty list.
        Does not handle cases with more than one categorical objective."""
        cm_dir = os.path.join(self.path(), CONFUSION_MATRIX_STR)
        return read_all_cms(cm_dir=cm_dir)
