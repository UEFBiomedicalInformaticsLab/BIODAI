from __future__ import annotations

from typing import Optional, Iterable, Sequence

from frozenlist import FrozenList
from numpy import ndarray
from pandas import DataFrame

from util.select_from_sequence import select_by_indices
from util.table.table_backend.table_backend import TableBackend
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table import Table


class BackedTable(Table):
    __backend: TableBackend
    __selected_rows: Optional[FrozenList[int]]
    __selected_cols: Optional[FrozenList[int]]
    """Read only class for representing generic 2D tables of floats that can be either in memory or backed by a file,
    with lazy operations, easily convertible to DataFrame or ndarray."""

    def __init__(self, backend: TableBackend,
                 selected_rows: Optional[Iterable[int]] = None, selected_cols: Optional[Iterable[int]] = None,
                 check_selections: bool = False):
        """Selections are safe-copied. Selected rows and cols are checked for range."""
        self.__backend = backend
        if selected_rows is None:
            self.__selected_rows = None
        else:
            self.__selected_rows = FrozenList(items=selected_rows)
            self.__selected_rows.freeze()
        if selected_cols is None:
            self.__selected_cols = None
        else:
            self.__selected_cols = FrozenList(items=selected_cols)
            self.__selected_cols.freeze()
        if check_selections:
            self.__check_selections()

    def __check_selections(self):
        if self.__selected_rows is not None:
            nrows = self.__backend.n_row()
            for r in self.__selected_rows:
                if not (0 <= r < nrows):
                    raise IndexError(
                        "Row selection out of range.\n" +
                        "Number of rows: " + str(nrows) + "\n"
                        "Selection:\n" +
                        str(self.__selected_rows) + "\n")
        if self.__selected_cols is not None:
            ncols = self.__backend.n_col()
            for c in self.__selected_cols:
                if not (0 <= c < ncols):
                    raise IndexError(
                        "Column selection out of range.\n" +
                        "Number of columns: " + str(ncols) + "\n"
                        "Selection:\n" +
                        str(self.__selected_cols) + "\n")

    def n_row(self) -> int:
        if self.__selected_rows is None:
            return self.__backend.n_row()
        else:
            return len(self.__selected_rows)

    def n_col(self) -> int:
        if self.__selected_cols is None:
            return self.__backend.n_col()
        else:
            return len(self.__selected_cols)

    def select_rows(self, selected: Iterable[int]) -> Table:
        """If an index is out of range throws an exception."""
        if self.__selected_rows is None:
            selected_new = selected
        else:
            selected_new = select_by_indices(data=self.__selected_rows, indices=selected)
        return BackedTable(backend=self.__backend, selected_rows=selected_new, selected_cols=self.__selected_cols)

    def select_cols(self, selected: Iterable[int]) -> Table:
        """If an index is out of range throws an exception."""
        if self.__selected_cols is None:
            selected_new = selected
        else:
            selected_new = select_by_indices(data=self.__selected_cols, indices=selected)
        return BackedTable(backend=self.__backend, selected_rows=self.__selected_rows, selected_cols=selected_new)

    def to_numpy(self) -> ndarray:
        """Returned object is either new or immutable."""
        return self.__backend.to_numpy(selected_rows=self.__selected_rows, selected_cols=self.__selected_cols)

    def inner_to_dataframe(self) -> DataFrame:
        return self.__backend.to_dataframe(selected_rows=self.__selected_rows, selected_cols=self.__selected_cols)

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        """Creates a faster access version, if reasonable. Can trade memory for speed, up to a specified
        memory expense. If the table is already fast (e.g. numpy backed) or too big, this same table is returned.
        Set max_cells to None to cache without limits."""
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            return BackedTable(backend=self.__backend.compile(
                selected_rows=self.__selected_rows, selected_cols=self.__selected_cols))

    def colnames(self) -> Sequence[str]:
        return self.__backend.colnames(selected=self.__selected_cols)

    def rownames(self) -> Sequence[str]:
        return self.__backend.rownames(selected=self.__selected_rows)

    def memory_size(self) -> int:
        return self.__backend.memory_size()

    def chunks_df(self, chunk_rows: Optional[int] = None
                  ) -> Iterable[DataFrame]:
        """Passes responsibility to the backend, allowing for faster specializations."""
        return self.__backend.chunk_iterable_df(
            chunk_rows=chunk_rows, selected_rows=self.__selected_rows, selected_cols=self.__selected_cols)

    def has_fast_cols(self) -> bool:
        return self.__backend.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__backend.has_fast_rows()
