from collections.abc import Sequence, Iterable
from typing import Optional

from numpy import ndarray
from pandas import DataFrame

from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_backend.np_table import NpTable


class TransposedTable(Table):
    """Wraps a Table transposing it."""
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def n_row(self) -> int:
        return self.__inner.n_col()

    def n_col(self) -> int:
        return self.__inner.n_row()

    def select_rows(self, selected: Iterable[int]) -> Table:
        return TransposedTable(table=self.__inner.select_cols(selected=selected))

    def select_cols(self, selected: Iterable[int]) -> Table:
        return TransposedTable(table=self.__inner.select_rows(selected=selected))

    def to_numpy(self) -> ndarray:
        return self.__inner.to_numpy().T

    def inner_to_dataframe(self) -> DataFrame:
        return self.__inner.inner_to_dataframe().T

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            from util.table.backed_table import BackedTable
            return BackedTable(
                backend=NpTable(data=self.to_numpy(), rownames=self.rownames(), colnames=self.colnames()))

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def colnames(self) -> Sequence[str]:
        return self.__inner.rownames()

    def rownames(self) -> Sequence[str]:
        return self.__inner.colnames()

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_rows()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_cols()
