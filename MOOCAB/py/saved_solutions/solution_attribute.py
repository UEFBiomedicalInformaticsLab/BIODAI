import os
from collections.abc import Sequence
from typing import Optional

import pandas as pd
from pandas import DataFrame

from consts import FINAL_STR
from cross_validation.multi_objective.cross_evaluator.hof_saver import SOLUTION_STR, SEP, FOLD_STR, CSV_EXTENSION, \
    INNER_CV_PREFIX, TEST_PREFIX
from plots.hof_utils import is_external_dir, select_and_remove_prefix, INTERNAL_CV_PREFIX, \
    INTERNAL_PREFIX, EXTERNAL_PREFIX
from util.dataframes import n_col
from util.named import NickNamed


def _fold_dfs_inner_cv_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INNER_CV_PREFIX)


def _fold_dfs_lasso_train_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    """Legacy code to read LASSO-MO performance on training set."""
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix="train")


def _fold_dfs_test_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=TEST_PREFIX)


def _train_data_from_dfs(dfs: Sequence[DataFrame]) -> Optional[Sequence[DataFrame]]:
    """Data from inner cv if available, from training on whole folds otherwise."""
    n_folds = len(dfs)
    if n_folds > 0:
        dfs_res = _fold_dfs_inner_cv_only(fold_dfs=dfs)
        if n_col(dfs_res[0]) > 0:
            return dfs_res
        else:
            dfs_res = _fold_dfs_lasso_train_only(fold_dfs=dfs)
            if n_col(dfs_res[0]) > 0:
                return dfs_res
    return None


def _test_data_from_dfs(dfs: Sequence[DataFrame]) -> Optional[Sequence[DataFrame]]:
    """Gets results for internal-external k-fold cross-validation. A df for each fold."""
    n_folds = len(dfs)
    if n_folds > 0:
        dfs = _fold_dfs_test_only(fold_dfs=dfs)
        n_objectives = n_col(dfs[0])
        if n_objectives > 0:
            return dfs
    return None


def _fold_dfs_internal_cv_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INTERNAL_CV_PREFIX)


def _fold_dfs_internal_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INTERNAL_PREFIX)


def _fold_dfs_external_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=EXTERNAL_PREFIX)


class SolutionAttribute(NickNamed):
    __nick: str
    __plural_nick: str

    def __init__(self, nick: str, plural_nick: str = None):
        self.__nick = nick
        if plural_nick is None:
            self.__plural_nick = nick
        else:
            self.__plural_nick = plural_nick

    def fold_file_prefix(self) -> str:
        return SOLUTION_STR + SEP + self.plural_nick() + SEP + FOLD_STR + SEP

    def nick(self) -> str:
        return self.__nick

    def plural_nick(self) -> str:
        return self.__plural_nick

    def read_fold_dfs(self, hof_dir: str) -> Sequence[DataFrame]:
        """Results from folds. Fold files are read in alphabetical order."""
        files = os.listdir(hof_dir)
        files.sort()
        dats = []
        for file in files:
            if file.startswith(self.fold_file_prefix()) and file.endswith(CSV_EXTENSION):
                if FINAL_STR not in file:
                    dats.append(pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file)))
        return dats

    def train_dfs(self, hof_dir: str) -> Optional[Sequence[DataFrame]]:
        """Data from inner cv if available, from training on whole folds otherwise.
        One df for each fold."""
        return _train_data_from_dfs(dfs=self.read_fold_dfs(hof_dir=hof_dir))

    def _test_cv_dfs(self, hof_dir: str) -> Optional[Sequence[DataFrame]]:
        """Gets results for internal-external k-fold cross-validation. A df for each fold."""
        return _test_data_from_dfs(dfs=self.read_fold_dfs(hof_dir=hof_dir))

    def to_test_dfs(self, hof_dir: str, verbose: str = False) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        dfs = None
        if os.path.isdir(hof_dir):
            if is_external_dir(hof_dir=hof_dir):
                dfs = [self.external_df(hof_dir=hof_dir)]
            else:
                dfs = self._test_cv_dfs(hof_dir=hof_dir)
            if dfs is None:
                print("Unable to create dataframes from directory " + str(hof_dir))
        else:
            if verbose:
                print("path is not a directory: " + str(hof_dir))
        return dfs

    def external_solution_file_name(self) -> str:
        # Used in external validation when there is just one solution set.
        return SOLUTION_STR + SEP + self.plural_nick() + CSV_EXTENSION

    def hof_external_solutions_df(self, hof_dir: str) -> DataFrame:
        for file in os.listdir(hof_dir):
            if file == self.external_solution_file_name():
                return pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
        raise ValueError("File not found in dir " + str(hof_dir) + "\n" +
                         "Expected file: " + str(self.external_solution_file_name()) + "\n" +
                         "Detected files:\n" + str([str(f) for f in os.listdir(hof_dir)]))

    def external_df(self, hof_dir: str) -> Optional[DataFrame]:
        df = self.hof_external_solutions_df(hof_dir=hof_dir)
        df = _fold_dfs_external_only(fold_dfs=[df])[0]
        n_objectives = n_col(df)
        if n_objectives > 0:
            return df
        return None

    def train_data(self, hof_dir: str, verbose: bool = False) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation.
        Uses the train cv results if they exist. Falls back to the results on all train set if they do not.
        If the hof is for internal-external CV, returns data from inner cv if available,
        from training on whole folds otherwise."""
        dfs = None
        if os.path.isdir(hof_dir):
            if is_external_dir(hof_dir=hof_dir):
                dfs = [self.internal_cv_df(hof_dir=hof_dir)]
            else:
                dfs = _train_data_from_dfs(dfs=self.read_fold_dfs(hof_dir=hof_dir))
            if verbose and dfs is None:
                print("Unable to create dataframes from directory " + str(hof_dir))
        else:
            print("path is not a directory: " + str(hof_dir))
        return dfs

    def test_combined_df(self, hof_dir: str) -> Optional[DataFrame]:
        dfs = self._test_cv_dfs(hof_dir=hof_dir)
        if dfs is not None:
            return pd.concat(dfs, ignore_index=True)
        else:
            return None

    def test_df(self, hof_dir: str) -> Optional[DataFrame]:
        """Puts all folds in one df."""
        df = self.test_combined_df(hof_dir=hof_dir)
        if df is None:
            return None
        n_objectives = n_col(df)
        if n_objectives > 1:
            return df
        else:
            return None

    def internal_cv_df(self, hof_dir: str) -> Optional[DataFrame]:
        """To get internal_cv attributes from external validation procedure. Falls back to attributes on whole train set
        if not available."""
        df = self.hof_external_solutions_df(hof_dir=hof_dir)
        df_res = _fold_dfs_internal_cv_only(fold_dfs=[df])[0]
        if n_col(df_res) > 1:
            return df_res
        else:
            df_res = _fold_dfs_internal_only(fold_dfs=[df])[0]
            if n_col(df_res) > 1:
                return df_res
            else:
                return None

    def to_final_internal_cv_df(self, hof_dir: str) -> Optional[DataFrame]:
        """Returns a df with the attributes computed on internal dataset if it is external validation,
        None otherwise."""
        df = None
        if os.path.isdir(hof_dir):
            if is_external_dir(hof_dir=hof_dir):
                df = self.internal_cv_df(hof_dir=hof_dir)
                if df is None:
                    print("Unable to create final dataframe from directory " + str(hof_dir))
            else:
                df = None  # Return none because we do not have the test attributes
        else:
            print("path is not a directory: " + str(hof_dir))
        return df

    def to_final_external_df(self, hof_dir: str) -> Optional[DataFrame]:
        """Returns a df with the fitnesses if it is external validation, None otherwise."""
        df = None
        if os.path.isdir(hof_dir):
            if is_external_dir(hof_dir=hof_dir):
                df = self.external_df(hof_dir=hof_dir)
                if df is None:
                    print("Unable to create final dataframe from directory " + str(hof_dir))
            else:
                df = None  # Return none because we do not have the test fitnesses
        else:
            print("path is not a directory: " + str(hof_dir))
        return df
