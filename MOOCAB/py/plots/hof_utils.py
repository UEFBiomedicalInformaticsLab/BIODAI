import os
from typing import Optional, Sequence

import pandas as pd
from pandas import DataFrame
from pandas.errors import EmptyDataError

from consts import FINAL_STR
from cross_validation.multi_objective.cross_evaluator.hof_saver import SOLUTION_FEATURES_PREFIX, \
    EXTERNAL_SOLUTION_FITNESSES_FILE_NAME, CSV_EXTENSION, SOLUTION_FITNESSES_STR
from external_validation.mo_external_evaluator.hof_saver import (SOLUTION_FEATURES_EXTERNAL, TRAIN_STR,
                                                                 INTERNAL_CV_STR, EXTERNAL_STR)
from util.dataframes import select_columns_by_prefix, remove_prefix_from_columns


EXTERNAL_PREFIX = EXTERNAL_STR + "_"
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
            if file.startswith(SOLUTION_FEATURES_PREFIX) and file.endswith(CSV_EXTENSION):
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
            if file.startswith(SOLUTION_FEATURES_PREFIX) and file.endswith(CSV_EXTENSION):
                if FINAL_STR not in file:
                    files.append(file)
    files.sort()
    res = []
    for f in files:
        joined = os.path.join(hof_dir, f)
        try:
            res.append(pd.read_csv(filepath_or_buffer=joined))
        except EmptyDataError as e:
            raise EmptyDataError("Error while reading " + str(joined) + ":\n" + str(e))
    return res


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
        if file == EXTERNAL_SOLUTION_FITNESSES_FILE_NAME:
            return True
    return False


def hof_final_fitness_df(hof_dir: str) -> Optional[DataFrame]:
    files = os.listdir(hof_dir)
    files.sort()
    res = None
    for file in files:
        if file.startswith(SOLUTION_FITNESSES_STR) and file.endswith(CSV_EXTENSION):
            if FINAL_STR in file:
                res = pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
    return res


def select_and_remove_prefix(fold_dfs: Sequence[DataFrame], prefix: str) -> Sequence[DataFrame]:
    dfs = [select_columns_by_prefix(df=df, prefix=prefix) for df in fold_dfs]
    for df in dfs:
        remove_prefix_from_columns(df=df, prefix=prefix)
    return dfs
