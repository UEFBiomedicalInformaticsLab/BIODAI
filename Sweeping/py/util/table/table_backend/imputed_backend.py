import math
from typing import Sequence, Optional

import numpy
from numpy import ndarray, ravel

from util.math.summer import KahanSummer
from util.table.table_backend.table_backend import TableBackend


class ImputedBackend(TableBackend):
    __inner: TableBackend
    __means: list[float]

    def __init__(self, backend: TableBackend):
        self.__inner = backend
        if self.__inner.n_row() == 0:
            self.__means = [math.nan for _ in range(self.__inner.n_col())]  # Not used anyway.
        self.__means = [KahanSummer.mean_unchecked(ravel(c), skip_nan=True) for c in backend.columns_df()]

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        res = self.__inner.to_new_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
        if selected_cols is None:
            selected_cols = range(self.n_col())
        means = self.__means
        for i, c in enumerate(selected_cols):
            res[numpy.isnan(res[:, i]), i] = means[c]
        return res

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        from util.table.table_backend.np_table import NpTable
        return NpTable(
            data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
            rownames=self.rownames(selected=selected_rows),
            colnames=self.colnames(selected=selected_cols))

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        return self.__inner.colnames(selected=selected)

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        return self.__inner.rownames(selected=selected)

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()
