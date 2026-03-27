from __future__ import annotations

from abc import ABC, abstractmethod
from math import isnan
from typing import Sequence, Iterable, Optional, Iterator

import numpy
import numpy as np
from numpy import ndarray, ravel
from pandas import DataFrame

from util.math.summer import KahanSummer
from util.math.utils import ceil_division
from util.progress_observer import ProgressObserverFactory, NULL_PROGRESS_OBSERVER_FACTORY, ProgressObserver
from util.sequence_utils import true_positions_sorted_set, true_positions
from util.table.table_backend.imputed_backend import ImputedBackend
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS, DEFAULT_CHUNK_CELLS, TABLE_DTYPE
from util.utils import IllegalStateError
from util.iterable_utils import SizedIterable


class Table(ABC):
    """Read only class for representing generic 2D tables of numbers that can be either in memory or backed by a file,
    with lazy operations, easily convertible to DataFrame or ndarray."""

    @abstractmethod
    def n_row(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def n_col(self) -> int:
        raise NotImplementedError()

    def size(self) -> int:
        return self.n_row() * self.n_col()

    @abstractmethod
    def select_rows(self, selected: Iterable[int]) -> Table:
        """Passed order and doubles are preserved."""
        raise NotImplementedError()

    @abstractmethod
    def select_cols(self, selected: Iterable[int]) -> Table:
        """Passed order and doubles are preserved."""
        raise NotImplementedError()

    def filter_rows_by_mask(self, mask: Sequence[bool]) -> Table:
        if len(mask) == self.n_row():
            return self.select_rows(selected=true_positions_sorted_set(s=mask))
        else:
            raise ValueError("Mask has wrong size.")

    def filter_cols_by_mask(self, mask: Sequence[bool]) -> Table:
        if len(mask) == self.n_col():
            return self.select_cols(selected=true_positions(s=mask))
        else:
            raise ValueError("Mask has wrong size. Table " + str(self.n_col()) + " vs mask " + str(len(mask)))

    @abstractmethod
    def to_numpy(self) -> ndarray:
        """Returned object is either new or immutable."""
        raise NotImplementedError()

    @abstractmethod
    def inner_to_dataframe(self) -> DataFrame:
        """Returned dataframe is a copy and can be modified."""
        raise NotImplementedError()

    def to_dataframe(self) -> DataFrame:
        """Returned dataframe is a copy and can be modified."""
        res = self.inner_to_dataframe()
        from util.table.table_utils import is_2d
        if is_2d(res):
            return res
        else:
            raise IllegalStateError("Generated dataframe is not 2D.\n" + str(res))

    @abstractmethod
    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        """Creates a faster access version, if reasonable. Can trade memory for speed, up to a specified
        memory expense. If the table is already fast (e.g. numpy backed) or too big, this same table is returned.
        Set max_cells to None to cache without limits."""
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self) -> int:
        """Number of cells stored in main memory."""
        raise NotImplementedError()

    def serialize(self) -> Table:
        """Returns self or another table with less memory occupation, to optimise message passing or storage space."""
        if self.size() <= self.memory_size():
            return self.compile(max_cells=None)
        else:
            return self

    @abstractmethod
    def colnames(self) -> Sequence[str]:
        raise NotImplementedError()

    @abstractmethod
    def rownames(self) -> Sequence[str]:
        raise NotImplementedError()

    def rows_in_range(self, selected: Iterable[int]) -> bool:
        """Checks if all passed row indices are in range."""
        size = self.n_row()
        for i in selected:
            if not (0 <= i < size):
                return False
        return True

    def cols_in_range(self, selected: Iterable[int]) -> bool:
        """Checks if all passed column indices are in range."""
        size = self.n_col()
        for i in selected:
            if not (0 <= i < size):
                return False
        return True

    def __str__(self) -> str:
        if self.size() <= 100:
            if self.size() == 0:
                return "Empty table"
            else:
                return str(self.to_dataframe())
        else:
            res = str(self.n_row()) + "*" + str(self.n_col()) + " table"
            res += "First rows and columns:\n"
            head = self.select_rows(selected=range(min(self.n_row(),5)))
            head = head.select_cols(selected=range(min(self.n_col(),5)))
            res += str(head.to_dataframe())
            return res

    def select_cols_by_names(self, names: Iterable[str]):
        """If two columns have the same name, the first one is returned."""
        col_names = self.colnames()
        return self.select_cols(selected=(col_names.index(n) for n in names))

    def select_rows_by_names(self, names: Sequence) -> Table:
        """If two rows have the same name, the first one is returned."""
        row_names = self.rownames()
        return self.select_rows(selected=(row_names.index(n) for n in names))

    def chunks(self, chunk_rows: Optional[int] = None) -> Iterable[Table]:
        """If chunk_rows is not specified, uses DEFAULT_MAX_CHUNK_CELLS / n_col, rounded up."""
        return ChunkIterable(table=self, chunk_rows=chunk_rows)

    def chunks_df(self, chunk_rows: Optional[int] = None) -> Iterable[DataFrame]:
        """If chunk_rows is not specified, uses DEFAULT_MAX_CHUNK_CELLS / n_col, rounded up.
        Can be overridden for faster specialisations."""
        return ChunkIterableDF(table=self, chunk_rows=chunk_rows)

    def rows_df(self) -> Iterable[DataFrame]:
        """Makes use of chunks_df that can be overridden for faster specializations."""
        return RowIterableDF(table=self)

    def n_cells(self) -> int:
        return self.n_row() * self.n_col()

    def shape(self) -> tuple[int, int]:
        return self.n_row(), self.n_col()

    def all_nan(self) -> bool:
        """Current version converts to numpy potentially wasting memory."""
        return np.isnan(self.to_numpy()).all()

    def remove_all_nan_cols(
            self, progress_observer_factory: ProgressObserverFactory = NULL_PROGRESS_OBSERVER_FACTORY) -> Table:
        """If nothing to remove returns self."""
        progress_observer = progress_observer_factory.create_progress_observer("Remove all NaN columns")
        progress_observer.notify_start()
        if self.has_fast_cols():
            progress_observer.notify_message(text="Fast columns detected, proceeding by column.")
            to_keep = self.__to_keep_nan_by_cols(progress_observer=progress_observer)
        else:
            progress_observer.notify_message(text="Fast columns not detected, proceeding by row.")
            to_keep = self.__to_keep_nan_by_rows(progress_observer=progress_observer)
        remaining_num = len(to_keep)
        tot_col = self.n_col()
        if remaining_num == self.n_col():
            res = self
            progress_observer.notify_end(report="0 columns removed.")
        else:
            res = self.select_cols(selected=to_keep)
            progress_observer.notify_end(report="Removed " + str(tot_col - remaining_num) + " columns out of " +
                                                str(tot_col) + ". Remaining: " + str(remaining_num) + ".")
        return res

    def __to_keep_nan_by_cols(
            self, progress_observer: ProgressObserver) -> Sequence[int]:
        to_keep = []
        tot_col = self.n_col()
        for i, c in enumerate(self.columns()):
            progress_observer.notify_progress(proportion=i/tot_col)
            if not c.all_nan():
                to_keep.append(i)
        return to_keep

    def __to_keep_nan_by_rows(
            self, progress_observer: ProgressObserver) -> Sequence[int]:
        tot_col = self.n_col()
        tot_row = self.n_row()
        to_keep = [False] * tot_col
        for i, r in enumerate(self.rows_df()):
            progress_observer.notify_progress(proportion=i/tot_row)
            row_values = r.iloc[0]  # Get the single row as a Series

            # Use vectorized logic to update `to_keep`
            to_keep = [keep or not np.isnan(val) for keep, val in zip(to_keep, row_values)]
        return [i for i, keep in enumerate(to_keep) if keep]

    def zero_var(self) -> bool:
        """Current version converts to numpy potentially wasting memory. Ignores NaN values.
        If all values are NaN returns False."""
        arr = self.to_numpy()
        res = np.nanmax(arr) - np.nanmin(arr) == 0
        return res

    def remove_zero_var_cols(
            self, progress_observer_factory: ProgressObserverFactory = NULL_PROGRESS_OBSERVER_FACTORY) -> Table:
        """Ignores NaN values, but removes the columns if all the values are NaN.
        If nothing to remove returns self."""
        table = self.remove_all_nan_cols(progress_observer_factory=progress_observer_factory)
        tot_col = table.n_col()
        progress_observer = progress_observer_factory.create_progress_observer("Remove 0 variance columns")
        progress_observer.notify_start()
        if self.has_fast_cols():
            progress_observer.notify_message(text="Fast columns detected, proceeding by column.")
            to_keep = self.__to_keep_var_by_cols(table=table, progress_observer=progress_observer)
        else:
            progress_observer.notify_message(text="Fast columns not detected, proceeding by row.")
            to_keep = self.__to_keep_var_by_rows(table=table, progress_observer=progress_observer)

        remaining_num = len(to_keep)
        if remaining_num == tot_col:
            res = table
            progress_observer.notify_end(report="0 columns removed.")
        else:
            res = table.select_cols(selected=to_keep)
            progress_observer.notify_end(report="Removed " + str(tot_col-remaining_num) + " columns out of " +
                                                str(tot_col) + ". Remaining: " + str(remaining_num) + ".")
        return res

    @staticmethod
    def __to_keep_var_by_cols(table: Table, progress_observer: ProgressObserver) -> Sequence[int]:
        to_keep = []
        tot_col = table.n_col()
        for i, c in enumerate(table.columns()):
            progress_observer.notify_progress(proportion=i / tot_col)
            if not c.zero_var():
                to_keep.append(i)
        return to_keep

    @staticmethod
    def __to_keep_var_by_rows(table: Table, progress_observer: ProgressObserver) -> Sequence[int]:
        """  Ignores NaN values. If all values are NaN the column is kept."""
        tot_col = table.n_col()
        tot_row = table.n_row()
        to_keep = [True] * tot_col
        values = [np.nan] * tot_col
        for i, r in enumerate(table.rows_df()):
            progress_observer.notify_progress(proportion=i / tot_row)
            row_values = r.iloc[0]  # Get the single row as a Series
            for j, val in enumerate(row_values):
                if np.isnan(val):
                    continue
                if np.isnan(values[j]):
                    values[j] = val
                    to_keep[j] = False
                elif val != values[j]:
                    to_keep[j] = True
        return [i for i, keep in enumerate(to_keep) if keep]

    def rows(self) -> Iterable[Table]:
        return RowIterable(table=self)

    def columns(self) -> Iterable[Table]:
        return ColIterable(table=self)

    def mean(self) -> float:
        """Current version converts to numpy potentially wasting memory. Ignores NaN values.
        Uses Kahan summer."""
        return KahanSummer.mean(e for e in self.to_numpy().ravel() if not isnan(e))

    def select_rows_cols(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> Table:
        table = self
        if selected_cols is not None:
            table = table.select_cols(selected=selected_cols)
        if selected_rows is not None:
            table = table.select_rows(selected=selected_rows)
        return table

    def impute(self) -> Table:
        """Imputes by column average. Columns of all NaN are removed."""
        from util.table.backed_table import BackedTable
        from util.table.table_backend.table_as_backend import TableAsBackend
        return BackedTable(backend=ImputedBackend(backend=TableAsBackend(table=self.remove_all_nan_cols())))

    def standardize(self) -> Table:
        from util.table.backed_table import BackedTable
        from util.table.table_backend.table_as_backend import TableAsBackend
        from util.table.table_backend.standardized_backend import StandardizedBackend
        return BackedTable(backend=StandardizedBackend(backend=TableAsBackend(table=self)))

    def n_missing(self) -> int:
        """Computed a column at a time to save memory."""
        sum_nan = 0
        for c in self.columns():
            sum_nan += np.isnan(c.to_numpy()).sum()
        return sum_nan

    def has_missing(self) -> int:
        """Computed a column at a time to save memory."""
        for c in self.columns():
            if np.isnan(c.to_numpy()).any():
                return True
        return False

    def rows_without_nan(self) -> Sequence[int]:
        res = []
        for i, r in enumerate(self.rows()):
            if not r.has_missing():
                res.append(i)
        return res

    def flatten(self) -> SizedIterable[TABLE_DTYPE]:
        """Values are returned column wise, going downward."""
        return FlattenIterable(table=self)

    def default_chunk_rows(self) -> int:
        return default_chunk_rows(num_col=self.n_col())

    def transpose(self) -> Table:
        from util.table.transposed_table import TransposedTable
        return TransposedTable(table=self)

    def __eq__(self, other) -> bool:
        """Equality is computed a column at a time to save memory."""
        if isinstance(other, Table):
            if self.n_row() == other.n_row() and self.n_col() == other.n_col():
                for c1, c2 in zip(self.columns(), other.columns()):
                    if not np.array_equal(c1.to_numpy(), c2.to_numpy()):
                        return False
                return True
            else:
                return False
        else:
            return False

    @abstractmethod
    def has_fast_cols(self) -> bool:
        """Either a file with fast random column access (e.g. HDF5 with vertical chunks) or data stored in memory
        (e.g. Numpy backed)."""
        raise NotImplementedError()

    @abstractmethod
    def has_fast_rows(self) -> bool:
        """Either a file with fast random row access (e.g. HDF5 with horizontal chunks) or data stored in memory
        (e.g. Numpy backed)."""
        raise NotImplementedError()

    def fast_cols(self) -> Table:
        """If it has fast columns returns itself. Otherwise, returns a compiled version. If the table is too big
        to compile still returns itself. It is possible to override to provide optimized specializations."""
        if self.has_fast_cols():
            return self
        else:
            return self.compile()

    def np_col(self, selected_col: int) -> ndarray:
        """Returns a 1-dimensional array."""
        return ravel(self.select_cols(selected=[selected_col]).to_numpy())

    def np_cols(self) -> Iterable[ndarray]:
        return (self.np_col(selected_col=i) for i in range(self.n_col()))

    def replace_column(self, new_column_pos: int, new_column: Sequence[float]) -> Table:
        from util.table.table_with_replaced_column import TableWithReplacedColumn
        return TableWithReplacedColumn(inner=self, new_column_pos=new_column_pos, new_column=new_column)

    def has_non_finite(self) -> bool:
        """Computed a column at a time to save memory."""
        for c in self.columns():
            if not numpy.isfinite(c.to_numpy()).all():
                return True
        return False


class ChunkIterator(Iterator[Table]):
    __inner: Table
    __chunk_rows: int
    __row: int

    def __init__(self, table: Table, chunk_rows: int):
        if chunk_rows < 1:
            raise ValueError()
        self.__inner = table
        self.__chunk_rows = chunk_rows
        self.__row = 0

    def __next__(self) -> Table:
        start_row = self.__row
        nrow = self.__inner.n_row()
        if start_row >= nrow:
            raise StopIteration()
        self.__row = start_row + self.__chunk_rows
        stop = min(self.__row, nrow)
        res = self.__inner.select_rows(selected=range(start_row, stop))
        return res


def default_chunk_rows(num_col: int) -> int:
    """DEFAULT_CHUNK_CELLS / n_col, rounded up."""
    return max(ceil_division(num=DEFAULT_CHUNK_CELLS, den=num_col), 1)


def clean_chunk_rows(num_col: int, chunk_rows: Optional[int] = None) -> int:
    """If chunk_rows is not specified, uses DEFAULT_CHUNK_CELLS / n_col, rounded up."""
    if chunk_rows is None:
        chunk_rows = default_chunk_rows(num_col=num_col)
    if chunk_rows < 1:
        raise ValueError()
    return chunk_rows


class ChunkIterable(Iterable[Table]):
    __inner: Table
    __chunk_rows: int

    def __init__(self, table: Table, chunk_rows: Optional[int] = None):
        """If chunk_rows is not specified, uses DEFAULT_CHUNK_CELLS / n_col, rounded up."""
        chunk_rows = clean_chunk_rows(num_col=table.n_col(), chunk_rows=chunk_rows)
        self.__inner = table
        self.__chunk_rows = chunk_rows

    def __iter__(self) -> Iterator[Table]:
        return ChunkIterator(table=self.__inner, chunk_rows=self.__chunk_rows)


class RowIteratorDF(Iterator[DataFrame]):
    __inner: Iterator[DataFrame]

    def __init__(self, inner: Iterator[DataFrame]):
        self.__inner = inner

    def __next__(self) -> DataFrame:
        return next(self.__inner)


class RowIterableDF(Iterable[DataFrame]):
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def __iter__(self) -> Iterator[DataFrame]:
        return RowIteratorDF(inner=iter(self.__inner.chunks_df(chunk_rows=1)))


class ColIterator(Iterator[Table]):
    __inner: Table
    __col: int

    def __init__(self, table: Table):
        self.__inner = table
        self.__col = 0

    def __next__(self) -> Table:
        if self.__col >= self.__inner.n_col():
            raise StopIteration()
        res = self.__inner.select_cols(selected=(self.__col,))
        self.__col += 1
        return res


class ColIterable(Iterable[Table]):
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def __iter__(self) -> Iterator[Table]:
        return ColIterator(table=self.__inner)


class RowIterator(Iterator[Table]):
    __inner: Table
    __row: int

    def __init__(self, table: Table):
        self.__inner = table
        self.__row = 0

    def __next__(self) -> Table:
        if self.__row >= self.__inner.n_row():
            raise StopIteration()
        res = self.__inner.select_rows(selected=(self.__row,))
        self.__row += 1
        return res


class RowIterable(Iterable[Table]):
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def __iter__(self) -> Iterator[Table]:
        return RowIterator(table=self.__inner)


class ChunkIteratorDF(Iterator[DataFrame]):
    __inner: Iterator[Table]

    def __init__(self, inner: Iterator[Table]):
        self.__inner = inner

    def __next__(self) -> DataFrame:
        return next(self.__inner).to_dataframe()


class ChunkIterableDF(Iterable[DataFrame]):
    __inner: Table
    __chunk_rows: int

    def __init__(self, table: Table, chunk_rows: Optional[int] = None):
        """If chunk_rows is not specified, uses DEFAULT_MAX_CHUNK_CELLS / n_col, rounded up."""
        chunk_rows = clean_chunk_rows(num_col=table.n_col(), chunk_rows=chunk_rows)
        self.__inner = table
        self.__chunk_rows = chunk_rows

    def __iter__(self) -> Iterator[DataFrame]:
        return ChunkIteratorDF(inner=ChunkIterator(table=self.__inner, chunk_rows=self.__chunk_rows))


class FlattenIterator(Iterator[TABLE_DTYPE]):
    __col_iterator: Iterator[Table]
    __cell_iterator: Iterator

    def __init__(self, table: Table):
        self.__col_iterator = iter(table.columns())
        self.__cell_iterator = np.nditer(next(self.__col_iterator).to_numpy())

    def __next__(self) -> TABLE_DTYPE:
        """Not recursive to save stack."""
        while True:
            try:
                return next(self.__cell_iterator)
            except StopIteration:
                self.__cell_iterator = np.nditer(next(self.__col_iterator).to_numpy())
                # A second stop iteration will be forwarded outside.


class FlattenIterable(SizedIterable[TABLE_DTYPE]):
    __inner: Table

    def __init__(self, table: Table):
        self.__inner = table

    def __iter__(self) -> Iterator[TABLE_DTYPE]:
        return FlattenIterator(table=self.__inner)

    def __len__(self) -> int:
        return self.__inner.size()