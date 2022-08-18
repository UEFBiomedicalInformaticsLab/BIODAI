import os
from typing import Optional, Sequence

import pandas as pd
from pandas import DataFrame

from util.dataframes import n_col, select_columns_by_prefix, remove_prefix_from_columns
from util.named import Named
from consts import FINAL_STR
from cross_validation.multi_objective.cross_evaluator.hof_saver import SOLUTION_FITNESSES_FILE_NAME, \
    SOLUTION_FITNESSES_FOLD_PREFIX, SOLUTION_FITNESSES_EXTENSION, TEST_PREFIX

EXTERNAL_PREFIX = "test_"


class SavedHoF(Named):
    __path: str
    __name: str

    def __init__(self, path: str, name: str):
        self.__path = path
        self.__name = name

    def path(self) -> str:
        return self.__path

    def name(self) -> str:
        return self.__name

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
        return to_dfs(hof_dir=self.path())


def is_external_dir(hof_dir: str) -> bool:
    for file in os.listdir(hof_dir):
        if file == SOLUTION_FITNESSES_FILE_NAME:
            return True
    return False


def test_df(hof_dir: str) -> Optional[DataFrame]:
    """Puts all folds in one df."""
    df = test_combined_df(hof_dir=hof_dir)
    if df is None:
        return None
    n_objectives = n_col(df)
    if n_objectives > 1:
        return df
    else:
        return None


def hof_fitness_dfs(hof_dir: str) -> Sequence[DataFrame]:
    """Fold files are read in alphabetical order."""
    files = os.listdir(hof_dir)
    files.sort()
    dats = []
    for file in files:
        if file.startswith(SOLUTION_FITNESSES_FOLD_PREFIX) and file.endswith(SOLUTION_FITNESSES_EXTENSION):
            if FINAL_STR not in file:
                dats.append(pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file)))
    return dats


def hof_final_fitness_df(hof_dir: str) -> Optional[DataFrame]:
    files = os.listdir(hof_dir)
    files.sort()
    res = None
    for file in files:
        if file.startswith(SOLUTION_FITNESSES_FOLD_PREFIX) and file.endswith(SOLUTION_FITNESSES_EXTENSION):
            if FINAL_STR in file:
                res = pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
    return res


def test_dfs(hof_dir: str) -> Optional[Sequence[DataFrame]]:
    dfs = hof_fitness_dfs(hof_dir=hof_dir)
    n_folds = len(dfs)
    if n_folds > 0:
        dfs = fold_dfs_test_only(fold_dfs=dfs)
        n_objectives = n_col(dfs[0])
        if n_objectives > 1:
            return dfs
    return None


def test_combined_df(hof_dir: str) -> Optional[DataFrame]:
    dfs = test_dfs(hof_dir=hof_dir)
    if dfs is not None:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None


def select_and_remove_prefix(fold_dfs: Sequence[DataFrame], prefix: str) -> Sequence[DataFrame]:
    dfs = [select_columns_by_prefix(df=df, prefix=prefix) for df in fold_dfs]
    for df in dfs:
        remove_prefix_from_columns(df=df, prefix=prefix)
    return dfs


def fold_dfs_test_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=TEST_PREFIX)


def fold_dfs_external_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=EXTERNAL_PREFIX)


def hof_solution_fitnesses_df(hof_dir: str) -> DataFrame:
    for file in os.listdir(hof_dir):
        if file == SOLUTION_FITNESSES_FILE_NAME:
            return pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
    raise ValueError("File not found in dir " + str(hof_dir) + "\n" +
                     "Expected file: " + str(SOLUTION_FITNESSES_FILE_NAME) + "\n" +
                     "Detected files:\n" + str([str(f) for f in os.listdir(hof_dir)]))


def external_df(hof_dir: str) -> Optional[DataFrame]:
    df = hof_solution_fitnesses_df(hof_dir=hof_dir)
    df = fold_dfs_external_only(fold_dfs=[df])[0]
    n_objectives = n_col(df)
    if n_objectives > 1:
        return df
    return None


def to_dfs(hof_dir: str) -> Optional[Sequence[DataFrame]]:
    """One df for each fold, or just a sequence of one df if it is from external validation."""
    dfs = None
    if os.path.isdir(hof_dir):
        if is_external_dir(hof_dir=hof_dir):
            dfs = [external_df(hof_dir=hof_dir)]
        else:
            dfs = test_dfs(hof_dir=hof_dir)
        if dfs is None:
            print("Unable to create dataframes from directory " + str(hof_dir))
    else:
        print("path is not a directory: " + str(hof_dir))
    return dfs


def to_final_df(hof_dir: str) -> Optional[DataFrame]:
    """Returns a df with the fitnesses if it is external validation, None otherwise."""
    df = None
    if os.path.isdir(hof_dir):
        if is_external_dir(hof_dir=hof_dir):
            df = external_df(hof_dir=hof_dir)
            if df is None:
                print("Unable to create final dataframe from directory " + str(hof_dir))
        else:
            df = None  # Return none because we do not have the test fitnesses
    else:
        print("path is not a directory: " + str(hof_dir))
    return df
