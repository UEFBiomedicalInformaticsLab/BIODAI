from typing import Optional, Sequence

from numpy import ndarray

from util.table.table import Table
from util.table.table_backend.table_backend import TableBackend


class TableAsBackend(TableBackend):
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        table = self.__inner.select_rows_cols(selected_rows=selected_rows, selected_cols=selected_cols)
        return table.to_numpy()

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        table = self.__inner.select_rows_cols(selected_rows=selected_rows, selected_cols=selected_cols)
        return TableAsBackend(table=table.compile())

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__inner.colnames()
        else:
            return self.__inner.select_cols(selected=selected).colnames()

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__inner.rownames()
        else:
            return self.__inner.select_rows(selected=selected).rownames()

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()
