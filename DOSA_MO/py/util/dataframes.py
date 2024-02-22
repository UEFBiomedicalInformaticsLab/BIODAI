import os
from collections.abc import Sequence, Iterable
from typing import Any

import numpy
import pandas as pd
from scipy.stats import pearsonr
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from util.math.sequences_to_float import SequencesToFloat


def has_nan(df):
    return df.isnull().values.any()


def nan_count(df):
    return df.isnull().values.sum()


def has_non_finite(df):
    return not numpy.isfinite(df).values.all()


def non_finite_report_unchecked(df: DataFrame) -> str:
    """Creates a verbose report. Call only if df contains unexpected values."""
    inf_num = inf_count(df)
    nan_num = nan_count(df)
    nan_s = nan_slice(df)
    err_msg = ""
    err_msg += "NaN or infinite value detected in data\n"
    err_msg += "Number of NaNs detected: " + str(nan_num) + "\n"
    err_msg += "Number of infinities detected: " + str(inf_num) + "\n"
    err_msg += "Data:\n"
    err_msg += str(df) + "\n"
    err_msg += "NaN slice:\n"
    err_msg += str(nan_s) + "\n"
    return err_msg


def has_non_finite_error(df: DataFrame) -> ValueError:
    """Creates a verbose error. Call only if df contains unexpected values."""
    return ValueError(non_finite_report_unchecked(df))


def non_finite_report(df: DataFrame) -> str:
    if has_non_finite(df=df):
        return non_finite_report_unchecked(df=df)
    else:
        return "NaN or infinite values not detected in data\n"


def has_inf(df):
    return numpy.isinf(df).values.any()


def inf_count(df):
    return numpy.isinf(df).values.sum()


def nan_slice(df):
    """ Slice of dataframe with only rows and columns containing NaN values. """
    is_nan = df.isna()
    return df.loc[is_nan.any(axis=1), is_nan.any(axis=0)]


def prefix_all_cols(df, prefix):
    return df.add_prefix(prefix)


def select_cols_by_mask(df: DataFrame, mask: Sequence[bool]) -> DataFrame:
    try:
        return df.loc[:, mask]
    except KeyError as e:
        raise KeyError("Passed mask size:" + str(len(mask)) + "\n" +
                       "Passed dataframe number of columns: " + str(n_col(df)) + "\n" +
                       "Original exception:\n" +
                       str(e) + "\n")


def standardize_df(df: DataFrame) -> DataFrame:
    scaled_features = StandardScaler().fit_transform(df.values)
    return DataFrame(scaled_features, index=df.index, columns=df.columns)


def scale_df(df: DataFrame, scaler) -> DataFrame:
    scaled_features = scaler.transform(df.values)
    return DataFrame(scaled_features, index=df.index, columns=df.columns)


def n_col(df: DataFrame) -> int:
    """Works also for numpy arrays"""
    return df.shape[1]


def n_row(df: DataFrame) -> int:
    """Works also for numpy arrays"""
    return df.shape[0]


def sum_by_columns(df: DataFrame) -> []:
    return df.sum(axis=0)


def create_from_labelled_lists(lists: [list]) -> DataFrame:
    """Fills with NaN"""
    dfs = []
    for li in lists:
        dfs.append(DataFrame([li]))
    return pd.concat(dfs, )


def columns_in_common(a: DataFrame, b: DataFrame) -> set:
    return set(a.columns).intersection(b.columns)


def select_columns_by_prefix(df: DataFrame, prefix: str) -> DataFrame:
    """Passed df is not modified."""
    filter_col = [col for col in df if col.startswith(prefix)]
    return df[filter_col]


def select_columns_by_suffix(df: DataFrame, suffix: str) -> DataFrame:
    """Passed df is not modified."""
    filter_col = [col for col in df if col.endswith(suffix)]
    return df[filter_col]


def remove_prefix_from_columns(df: DataFrame, prefix: str):
    """Passed df is modified in place. Does not modify column names not beginning with passed prefix."""
    df.columns = [c.removeprefix(prefix) for c in df.columns]


def create_df_with_repeated_value(value: Any, height: int, width: int) -> DataFrame:
    return pd.DataFrame(value, index=range(height), columns=range(width))


def cbind(dfs: Iterable[DataFrame]) -> DataFrame:
    return pd.concat(dfs, axis=1)


def replace_column_by_iat(df: DataFrame, col_pos: int, col_data: Sequence):
    """iat is slow, but trying to assign the whole column in one call produces warnings."""
    n = n_row(df)
    for i in range(n):
        new_val = col_data[i]
        df.iat[i, col_pos] = new_val


def replace_column_by_squares(df: DataFrame, col_pos: int, col_data: Sequence):
    """To assign the whole column we need to disable warnings."""
    with pd.option_context('mode.chained_assignment', None):
        df[df.columns[col_pos]] = col_data


def replace_column(df: DataFrame, col_pos: int, col_data: Sequence):
    replace_column_by_squares(df=df, col_pos=col_pos, col_data=col_data)


def to_csv_makingdirs(df: DataFrame, path: str, index: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path_or_buf=path, index=index)


def select_by_row_indices(samples: DataFrame, indices) -> DataFrame:
    """Uses actual locations, not row names."""
    return samples.iloc[indices]


def row_as_list(df: DataFrame, row: int) -> list:
    return df.iloc[row, :].values.flatten().tolist()


def checked_df_subtraction(df1: DataFrame, df2: DataFrame) -> DataFrame:
    if df1.shape == df2.shape:
        return df1.subtract(df2)
    else:
        raise ValueError()


def columnwise_correlations(df1: DataFrame, df2: DataFrame, corr_function=pearsonr) -> list[float]:
    if df1.shape == df2.shape:
        return [corr_function(df1.iloc[:, i], df2.iloc[:, i])[0] for i in range(n_col(df1))]
    else:
        raise ValueError()


def columnwise_correlations_p_val(df1: DataFrame, df2: DataFrame, corr_function=pearsonr) -> list[float]:
    if df1.shape == df2.shape:
        return [corr_function(df1.iloc[:, i], df2.iloc[:, i])[1] for i in range(n_col(df1))]
    else:
        raise ValueError()


def columnwise_measures(df1: DataFrame, df2: DataFrame, measure: SequencesToFloat) -> list[float]:
    if df1.shape == df2.shape:
        return [measure.apply(seq1=df1.iloc[:, i], seq2=df2.iloc[:, i]) for i in range(n_col(df1))]
    else:
        raise ValueError()


def has_column(df: DataFrame, col_name: str) -> bool:
    return col_name in df.columns
