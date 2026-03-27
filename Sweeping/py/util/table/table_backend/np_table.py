from __future__ import annotations
from typing import Sequence, Union, Optional

import numpy
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from static_frame import Frame
from frozenlist import FrozenList

from util.table.table_backend.table_backend import TableBackend
from util.table.table_utils import is_2d, is_table_data, n_row, n_col


class NpTable(TableBackend):
    """Array, rownames and colnames are unmodifiable."""
    __data: ndarray
    __is_memory_mapped: bool
    __rownames: FrozenList[str]
    __colnames: FrozenList[str]

    def __init__(self, data: Union[ndarray, DataFrame, Frame, str],
                 rownames: Optional[Sequence[str]] = None, colnames: Optional[Sequence[str]] = None):
        """Passed array is safe-copied. Internal copy is made non-writable and views of it will be returned
        to save memory. If rownames are specified they are used, otherwise the row names of the data are
        used if present. The same for colnames.
        If data is a string it will be interpreted as a path to an array stored in a file, and the file will
        be opened in read only mode and be memory mapped. When compiling, a copy in memory is created."""
        if isinstance(data, ndarray):
            data = data.copy()
        if isinstance(data, str):
            data = np.load(file=data, mmap_mode='r', allow_pickle=False)
            self.__is_memory_mapped = True
        else:
            self.__is_memory_mapped = False
        if not is_2d(data):
            raise ValueError("data is not 2D")
        if not is_table_data(data):
            raise ValueError("data is not numeric")
        assigned_rownames = False
        assigned_colnames = False
        if rownames is not None:
            self.__rownames = FrozenList(items=(str(n) for n in rownames))
            assigned_rownames = True
        if colnames is not None:
            self.__colnames = FrozenList(items=(str(n) for n in colnames))
            assigned_colnames = True
        if isinstance(data, ndarray):
            self.__data = data
            if not assigned_rownames:
                self.__rownames = FrozenList(items=(str(i) for i in range(n_row(self.__data))))
            if not assigned_colnames:
                self.__colnames = FrozenList(items=(str(i) for i in range(n_col(self.__data))))
        elif isinstance(data, DataFrame):
            self.__data = data.to_numpy(copy=True)
            if not assigned_rownames:
                self.__rownames = FrozenList(items=(str(n) for n in data.index))
            if not assigned_colnames:
                self.__colnames = FrozenList(items=(str(n) for n in data.columns))
        elif isinstance(data, Frame):
            self.__data = data.values
            if not assigned_rownames:
                self.__rownames = FrozenList(items=(str(n) for n in data.index))
            if not assigned_colnames:
                self.__colnames = FrozenList(items=(str(n) for n in data.columns))
        else:
            raise ValueError("Unsupported input type.")
        self.__data.setflags(write=False)
        self.__rownames.freeze()
        self.__colnames.freeze()

    def n_row(self) -> int:
        return self.__data.shape[0]

    def n_col(self) -> int:
        return self.__data.shape[1]

    @staticmethod
    def select_from_numpy(
            data: ndarray, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        if selected_rows is None:
            if selected_cols is None:
                return data
            else:
                return data[:, selected_cols]
        else:
            if selected_cols is None:
                return data[selected_rows, :]
            else:
                return data[np.ix_(selected_rows, selected_cols)]

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        if selected_rows is None:
            if selected_cols is None:
                return self.__data
            else:
                return self.__data[:, selected_cols]
        else:
            if selected_cols is None:
                return self.__data[selected_rows, :]
            else:
                return self.__data[np.ix_(selected_rows, selected_cols)]

    def to_frame(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> Frame:
        return Frame(data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
                     index=self.rownames(selected=selected_rows), columns=self.colnames(selected=selected_cols))

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        if selected_rows is None and selected_cols is None and not isinstance(self.__data, np.memmap):
            return self
        else:
            data = self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
            if isinstance(data, np.memmap):
                data = numpy.copy(data)  # If reading from disk we make a faster copy in memory.
            return NpTable(
                data=data,
                rownames=self.rownames(selected=selected_rows),
                colnames=self.colnames(selected=selected_cols))

    def memory_size(self) -> int:
        if isinstance(self.__data, np.memmap):
            return 0
        else:
            return self.size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__colnames
        else:
            return [self.__colnames[s] for s in selected]

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__rownames
        else:
            return [self.__rownames[s] for s in selected]

    def has_fast_cols(self) -> bool:
        return not self.__is_memory_mapped

    def has_fast_rows(self) -> bool:
        """Maybe we could return True also for memory mapped arrays. Benchmarking vs HDF5 is needed."""
        return not self.__is_memory_mapped