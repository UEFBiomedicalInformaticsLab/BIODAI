from __future__ import annotations

from typing import Union

import numpy
from numpy import ndarray, issubdtype, number
from pandas import DataFrame
from static_frame import Frame
from util.table.table_consts import TABLE_DTYPE
from util.table.table import Table



def is_2d(data: Union[DataFrame, numpy.ndarray, Frame, Table]) -> bool:
    if isinstance(data, Table):
        return True
    else:
        return len(data.shape) == 2


def n_col(data: Union[DataFrame, ndarray, Table, Frame]) -> int:
    if isinstance(data, Table):
        return data.n_col()
    else:
        return data.shape[1]


def n_row(data: Union[DataFrame, ndarray, Table, Frame]) -> int:
    if isinstance(data, Table):
        return data.n_row()
    else:
        return data.shape[0]


def is_numeric_data(data: Union[DataFrame, ndarray, Table, Frame]) -> bool:
    if isinstance(data, Table):
        return True
    if isinstance(data, ndarray):
        return issubdtype(data.dtype, number)
    if isinstance(data, DataFrame):
        return data.select_dtypes(include=[number]).shape[1] == data.shape[1]
    if isinstance(data, Frame):
        return all(issubdtype(dtype, number) for dtype in data.dtypes.values)
    else:
        raise ValueError()


def is_table_data(data: Union[DataFrame, ndarray, Table, Frame]) -> bool:
    if isinstance(data, Table):
        return True
    if isinstance(data, ndarray):
        return issubdtype(data.dtype, TABLE_DTYPE)
    if isinstance(data, DataFrame):
        return data.select_dtypes(include=[TABLE_DTYPE]).shape[1] == data.shape[1]
    if isinstance(data, Frame):
        return all(issubdtype(dtype, TABLE_DTYPE) for dtype in data.dtypes.values)
    else:
        raise ValueError()
