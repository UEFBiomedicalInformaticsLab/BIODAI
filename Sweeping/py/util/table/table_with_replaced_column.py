from typing import Sequence, Optional, Iterable

from numpy import ndarray, array, fromiter, intp
from pandas import DataFrame

from util.dataframe.dataframes import replace_column
from util.iterable_utils import indices_of
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS


class TableWithReplacedColumn(Table):
    __inner: Table
    __new_column_pos: int
    __new_column: ndarray

    def __init__(self, inner: Table, new_column_pos: int, new_column: Sequence[float]):
        """The passed new column is safe copied."""
        if new_column_pos < 0 or new_column_pos >= inner.n_col():
            raise IndexError()
        self.__inner = inner
        self.__new_column_pos = new_column_pos
        self.__new_column = array(new_column, dtype=float)
        if len(self.__new_column) != inner.n_row():
            raise ValueError()

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    def select_rows(self, selected: Iterable[int]) -> Table:
        res = self.__inner.select_rows(selected=selected)
        idx = fromiter(selected, dtype=intp)
        res = TableWithReplacedColumn(
            inner=res, new_column_pos=self.__new_column_pos, new_column=self.__new_column[idx])
        return res

    def select_cols(self, selected: Iterable[int]) -> Table:
        res = self.__inner.select_cols(selected=selected)
        for i in indices_of(iterable=selected, target=self.__new_column_pos):
            res = TableWithReplacedColumn(inner=res, new_column=self.__new_column, new_column_pos=i)
        return res

    def to_numpy(self) -> ndarray:
        res = self.__inner.to_numpy()

        # Make a writable copy if needed
        if not getattr(res, "flags", None) or not res.flags.writeable:
            res = res.copy()  # allocates new buffer, typically writable

        res[:, self.__new_column_pos] = self.__new_column
        return res

    def inner_to_dataframe(self) -> DataFrame:
        res = self.__inner.inner_to_dataframe()
        replace_column(df=res, col_pos=self.__new_column_pos, col_data=self.__new_column)
        return res

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            from util.table.table_backend.np_table import NpTable
            return BackedTable(
                backend=NpTable(data=self.to_numpy(), rownames=self.rownames(), colnames=self.colnames()))

    def memory_size(self) -> int:
        return self.__inner.memory_size() + self.n_col()

    def colnames(self) -> Sequence[str]:
        return self.__inner.colnames()

    def rownames(self) -> Sequence[str]:
        return self.__inner.rownames()

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()
