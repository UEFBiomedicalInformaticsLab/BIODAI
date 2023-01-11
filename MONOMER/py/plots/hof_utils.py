import os
from typing import Optional, Sequence

import pandas as pd
from pandas import DataFrame

from consts import FINAL_STR
from cross_validation.multi_objective.cross_evaluator.hof_saver import SOLUTION_FEATURES_PREFIX, \
    SOLUTION_FEATURES_EXTENSION, SOLUTION_FITNESSES_FILE_NAME, SOLUTION_FITNESSES_FOLD_PREFIX, \
    SOLUTION_FITNESSES_EXTENSION, INNER_CV_PREFIX, TEST_PREFIX
from external_validation.mo_external_evaluator.hof_saver import SOLUTION_FEATURES_EXTERNAL, TRAIN_STR, INTERNAL_CV_STR
from util.dataframes import n_col, select_columns_by_prefix, remove_prefix_from_columns


EXTERNAL_PREFIX = "test_"
INTERNAL_PREFIX = TRAIN_STR + "_"
INTERNAL_CV_PREFIX = INTERNAL_CV_STR + "_"


def hof_used_features_final_df(hof_dir: str) -> Optional[DataFrame]:
    """Works for both internal final training and external validation. Returns None if nothing to read."""
    file_to_read = None
    if is_external_dir(hof_dir=hof_dir):
        for file in os.listdir(hof_dir):
            if file == SOLUTION_FEATURES_EXTERNAL:
                file_to_read = file
    else:
        for file in os.listdir(hof_dir):
            if file.startswith(SOLUTION_FEATURES_PREFIX) and file.endswith(SOLUTION_FEATURES_EXTENSION):
                if FINAL_STR in file:
                    file_to_read = file
    if file_to_read is None:
        return None
    else:
        return pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file_to_read))


def hof_used_features_fold_dfs(hof_dir: str) -> Sequence[DataFrame]:
    """Fold dfs are returned in alphabetic order"""
    files = []
    if is_external_dir(hof_dir=hof_dir):
        for file in os.listdir(hof_dir):
            if file == SOLUTION_FEATURES_EXTERNAL:
                files.append(file)
    else:
        for file in os.listdir(hof_dir):
            if file.startswith(SOLUTION_FEATURES_PREFIX) and file.endswith(SOLUTION_FEATURES_EXTENSION):
                if FINAL_STR not in file:
                    files.append(file)
    files.sort()
    return [pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, f)) for f in files]


def feature_counts_from_df(df: DataFrame) -> Sequence[int]:
    return [sum(row) for row in df.values.tolist()]


def hof_folds_feature_counts(hof_dir: str) -> Sequence[Sequence[int]]:
    """Folds are returned in alphabetic order. For each fold returns a sequence of feature counts for the solutions."""
    dfs = hof_used_features_fold_dfs(hof_dir=hof_dir)
    res = []
    for df in dfs:
        res.append(feature_counts_from_df(df=df))
    return res


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


def train_cv_dfs(hof_dir: str) -> Optional[Sequence[DataFrame]]:
    dfs = hof_fitness_dfs(hof_dir=hof_dir)
    n_folds = len(dfs)
    if n_folds > 0:
        dfs_res = fold_dfs_inner_cv_only(fold_dfs=dfs)
        if n_col(dfs_res[0]) > 1:
            return dfs_res
        else:
            dfs_res = fold_dfs_lasso_train_only(fold_dfs=dfs)
            if n_col(dfs_res[0]) > 1:
                return dfs_res
    return None


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


def fold_dfs_lasso_train_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    """Legacy code to read LASSO-MO performance on training set."""
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix="train")


def fold_dfs_inner_cv_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INNER_CV_PREFIX)


def fold_dfs_test_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=TEST_PREFIX)


def fold_dfs_internal_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INTERNAL_PREFIX)


def fold_dfs_internal_cv_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=INTERNAL_CV_PREFIX)


def fold_dfs_external_only(fold_dfs: Sequence[DataFrame]) -> Sequence[DataFrame]:
    return select_and_remove_prefix(fold_dfs=fold_dfs, prefix=EXTERNAL_PREFIX)


def hof_solution_fitnesses_df(hof_dir: str) -> DataFrame:
    for file in os.listdir(hof_dir):
        if file == SOLUTION_FITNESSES_FILE_NAME:
            return pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
    raise ValueError("File not found in dir " + str(hof_dir) + "\n" +
                     "Expected file: " + str(SOLUTION_FITNESSES_FILE_NAME) + "\n" +
                     "Detected files:\n" + str([str(f) for f in os.listdir(hof_dir)]))


def internal_cv_df(hof_dir: str) -> Optional[DataFrame]:
    """To get internal_cv fitnesses from external validation procedure. Falls back to fitnesses on whole train set
    if not available."""
    df = hof_solution_fitnesses_df(hof_dir=hof_dir)
    df_res = fold_dfs_internal_cv_only(fold_dfs=[df])[0]
    if n_col(df_res) > 1:
        return df_res
    else:
        df_res = fold_dfs_internal_only(fold_dfs=[df])[0]
        if n_col(df_res) > 1:
            return df_res
        else:
            return None


def external_df(hof_dir: str) -> Optional[DataFrame]:
    df = hof_solution_fitnesses_df(hof_dir=hof_dir)
    df = fold_dfs_external_only(fold_dfs=[df])[0]
    n_objectives = n_col(df)
    if n_objectives > 1:
        return df
    return None


def to_train_dfs(hof_dir: str) -> Optional[Sequence[DataFrame]]:
    """One df for each fold, or just a sequence of one df if it is from external validation.
    Uses the train cv results if they exist. Falls back to the results on all train set if they do not."""
    dfs = None
    if os.path.isdir(hof_dir):
        if is_external_dir(hof_dir=hof_dir):
            dfs = [internal_cv_df(hof_dir=hof_dir)]
        else:
            dfs = train_cv_dfs(hof_dir=hof_dir)
        if dfs is None:
            print("Unable to create dataframes from directory " + str(hof_dir))
    else:
        print("path is not a directory: " + str(hof_dir))
    return dfs


def to_test_dfs(hof_dir: str) -> Optional[Sequence[DataFrame]]:
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


def to_final_internal_cv_df(hof_dir: str) -> Optional[DataFrame]:
    """Returns a df with the fitnesses if it is external validation, None otherwise."""
    df = None
    if os.path.isdir(hof_dir):
        if is_external_dir(hof_dir=hof_dir):
            df = internal_cv_df(hof_dir=hof_dir)
            if df is None:
                print("Unable to create final dataframe from directory " + str(hof_dir))
        else:
            df = None  # Return none because we do not have the test fitnesses
    else:
        print("path is not a directory: " + str(hof_dir))
    return df


def to_final_external_df(hof_dir: str) -> Optional[DataFrame]:
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
