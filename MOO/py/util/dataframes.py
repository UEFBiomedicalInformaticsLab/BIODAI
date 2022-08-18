import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


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


def select_cols_by_mask(df, mask: list[bool]):
    return df.loc[:, mask]


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
