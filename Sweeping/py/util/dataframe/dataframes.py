import os
import time
from collections.abc import Sequence, Iterable
from typing import Any, Union

import numpy
import numpy as np
import pandas as pd
from pandas.errors import IndexingError
from scipy.stats import pearsonr
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from static_frame import Frame
import pandas.api.types as ptypes

from util.math.sequences_to_float import SequencesToFloat
from util.sparse_bool_list_by_set import SparseBoolList
from util.table.table_utils import is_2d, n_row, n_col


MAX_CATEGORIES = 30
"""Max categories when automatically choosing to use one-hot encoding."""


def has_nan(df):
    return df.isnull().values.any()


def nan_count(df):
    return df.isnull().values.sum()


def has_non_finite(data: Union[DataFrame, numpy.ndarray, Frame]):
    if isinstance(data, (DataFrame, Frame)):
        return not numpy.isfinite(data.values).all()
    else:
        return not numpy.isfinite(data).all()


def non_finite_report_unchecked(df: Union[DataFrame, Frame]) -> str:
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


def has_non_finite_error(df: Union[DataFrame, Frame]) -> ValueError:
    """Creates a verbose error. Call only if df contains unexpected values."""
    return ValueError(non_finite_report_unchecked(df))


def non_finite_report(df: DataFrame) -> str:
    if has_non_finite(data=df):
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
    """Raises exception if mask has wrong length."""
    if not is_2d(df):
        raise ValueError("DataFrame is not 2D.\n" + "Passed dataframe shape: " + str(df.shape) + "\n")
    if len(mask) != n_col(df):
        raise ValueError("Mask has wrong length." +
                         "Passed mask size:" + str(len(mask)) + "\n" +
                         "Passed dataframe shape: " + str(df.shape) + "\n")
    if isinstance(mask, SparseBoolList):
        return df.iloc[:, mask.true_positions()]
    try:
        return df.loc[:, mask]
    except (KeyError, IndexingError) as e:
        raise KeyError("Passed mask size:" + str(len(mask)) + "\n" +
                       "Passed dataframe shape: " + str(df.shape) + "\n" +
                       "Original exception:\n" +
                       str(e) + "\n")


def standardize_df(df: DataFrame) -> DataFrame:
    scaled_features = StandardScaler().fit_transform(df.values)
    return DataFrame(scaled_features, index=df.index, columns=df.columns)


def scale_df(df: DataFrame, scaler) -> DataFrame:
    scaled_features = scaler.transform(df.values)
    return DataFrame(scaled_features, index=df.index, columns=df.columns)


def sum_by_columns(df: DataFrame) -> Sequence:
    return df.sum(axis=0)


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
    """iat is slower than assigning the whole column in one call."""
    n = n_row(df)
    for i in range(n):
        new_val = col_data[i]
        df.iat[i, col_pos] = new_val


def replace_column_by_squares(df: DataFrame, col_pos: int, col_data: Sequence):
    """Replaces in place."""
    # To assign the whole column we need to disable warnings.
    with pd.option_context('mode.chained_assignment', None):
        df[df.columns[col_pos]] = col_data


def replace_column(df: DataFrame, col_pos: int, col_data: Sequence):
    """Replaces in place."""
    replace_column_by_squares(df=df, col_pos=col_pos, col_data=col_data)


def to_csv_makingdirs(df: DataFrame, path: str, index: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path_or_buf=path, index=index)


def select_by_row_indices(samples: DataFrame, indices: Sequence[int]) -> DataFrame:
    """Uses actual locations, not row names."""
    return samples.iloc[indices]


def select_by_row_names(samples: DataFrame, names) -> DataFrame:
    """Uses actual locations, not row names."""
    return samples.loc[names]


def row_as_list(df: DataFrame, row: int) -> list:
    num_rows = n_row(df)
    if not (0 <= row < num_rows):
        raise IndexError("Requested row " + str(row) + " of a dataframe with " + str(num_rows) + " rows.")
    else:
        return df.iloc[row, :].values.flatten().tolist()


def col_as_list(df: DataFrame, col: int) -> list:
    num_cols = n_col(df)
    if not (0 <= col < num_cols):
        raise IndexError("Requested col " + str(col) + " of a dataframe with " + str(num_cols) + " cols.")
    else:
        return df.iloc[:, col].values.flatten().tolist()


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


def common_row_names(dfs: Sequence[DataFrame]) -> set[str]:
    first_df = True
    res = {}
    for d in dfs:
        d_index = d.index
        d_unique = set(d_index)
        if len(d_index) != len(d_unique):
            raise ValueError("row names are not unique in one of the dataframes.")
        if first_df:
            res = d_unique
            first_df = False
        else:
            res = res.intersection(d_unique)
    return res


def has_negatives(df: DataFrame) -> bool:
    return bool((df.values < 0).any())


def integer_cols_when_possible(df: DataFrame) -> DataFrame:
    df = df.copy()
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Convert numeric columns to integers if they do not have a decimal part
    for col in numeric_cols:
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)

    return df


def should_one_hot_encode(df: DataFrame, feature) -> bool:
    if not ptypes.is_numeric_dtype(df[feature]):
        return True
    # Check if the feature is categorical
    unique_values = df[feature].nunique()
    if 2 < unique_values <= MAX_CATEGORIES:  # Arbitrary threshold for categorical features
        return True
    return False


def encode_features(df: DataFrame) -> DataFrame:
    """Creates a new DataFrame encoding features as one-hot when the feature seems well suited for that
    representation."""
    encoded_df = df.copy()
    for feature in df.columns:
        if should_one_hot_encode(df, feature):
            # Apply one-hot encoding
            one_hot = pd.get_dummies(df[feature], prefix=feature)
            one_hot = one_hot.astype(int)
            encoded_df = encoded_df.drop(feature, axis=1)
            encoded_df = pd.concat([encoded_df, one_hot], axis=1)
    return encoded_df


def substitute_columns_by_sum(df: DataFrame, col_a: str, col_b: str, col_sum: str) -> DataFrame:
    df = df.copy()
    df[col_sum] = df[col_a] + df[col_b]
    if col_a != col_sum:
        df.drop(columns=[col_a], inplace=True)
    if col_b != col_sum:
        df.drop(columns=[col_b], inplace=True)
    return df


def filter_df2_by_df1_index_locally(
        index_df: pd.DataFrame, target_df: pd.DataFrame, verbose: bool = True):
    """
    Filters and reorders target_df in place to match index_df's index, using minimal memory.
    Prints progress if verbose is True.
    """

    # Step 1: Keep only rows in df2 that have matching index in df1
    matching_index = target_df.index.intersection(index_df.index)
    if verbose:
        print("Dropping rows without matching index.")
    target_df.drop(index=target_df.index.difference(matching_index), inplace=True)

    # Step 2: Compute the permutation to match df1's index order
    indexer = index_df.index.get_indexer(target_df.index)

    batch_size = max(3000000 // n_row(target_df), 1)  # Seems to be a reasonable sweet spot after some tests.

    if verbose:
        print("Reordering target dataframe in place, a batch of columns at a time.")
    columns = target_df.columns
    start_time = time.time()  # Start timing

    for i, start in enumerate(range(0, len(columns), batch_size)):
        end = min(start + batch_size, len(columns))
        batch_cols = columns[start:end]
        batch_data = target_df[batch_cols].values  # shape: (n_rows, batch_size)

        reordered_batch = np.empty((len(index_df.index), len(batch_cols)), dtype=batch_data.dtype)
        reordered_batch[indexer, :] = batch_data

        target_df[batch_cols].values[:] = reordered_batch
        if verbose and i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Reordered columns {start + 1} to {end} out of {len(columns)} "
                  f"(elapsed time: {elapsed / 60:.2f} minutes)")

    if verbose:
        elapsed = time.time() - start_time
        print(f"All the columns have been reordered (elapsed time: {elapsed / 60:.2f} minutes)")

    # Step 4: Set target's index to index_df's index
    target_df.index = index_df.index
