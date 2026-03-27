from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Iterable, Optional, Sequence

import numpy
from numpy import ndarray

from pandas import DataFrame, Index


class TableBackend(ABC):
    """Differently from a Table, a TableBackend does not contain a representation of high level operations like
    selection or composition."""

    @abstractmethod
    def n_row(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def n_col(self) -> int:
        raise NotImplementedError()

    def size(self) -> int:
        return self.n_row() * self.n_col()

    @abstractmethod
    def  to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        """Returned object is either new or immutable."""
        raise NotImplementedError()

    def to_new_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        """Returned object is guaranteed to be new and modifiable."""
        res = self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
        if not res.flags.writeable:
            res = numpy.copy(res)
        return res

    def to_dataframe(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> DataFrame:
        return DataFrame(data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
                         index=Index(self.rownames(selected=selected_rows)),
                         columns=Index(self.colnames(selected=selected_cols)),
                         copy=True, dtype=float)

    @abstractmethod
    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self) -> int:
        """Number of cells in main memory."""
        raise NotImplementedError()

    @abstractmethod
    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        raise NotImplementedError()

    @abstractmethod
    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        raise NotImplementedError()

    def cols_in_range(self, selected: Iterable[int]) -> bool:
        size = self.n_col()
        for i in selected:
            if i < 0 or i >= size:
                return False
        return True

    def chunk_iterable_df(self, chunk_rows: Optional[int],
                          selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]
                          ) -> Iterable[DataFrame]:
        """This can be overridden for faster specializations."""
        from util.table.backed_table import BackedTable
        from util.table.table import ChunkIterableDF
        return ChunkIterableDF(
            table=BackedTable(
                backend=self, selected_rows=selected_rows, selected_cols=selected_cols), chunk_rows=chunk_rows)

    def columns_df(self) -> ColIteratorDF:
        return ColIteratorDF(self)

    @abstractmethod
    def has_fast_cols(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_fast_rows(self) -> bool:
        raise NotImplementedError()


class ColIteratorDF(Iterator[DataFrame]):
    __inner: TableBackend
    __col: int

    def __init__(self, backend: TableBackend):
        self.__inner = backend
        self.__col = 0

    def __next__(self) -> DataFrame:
        if self.__col >= self.__inner.n_col():
            raise StopIteration()
        res = self.__inner.to_dataframe(selected_rows=None, selected_cols=(self.__col,))
        self.__col += 1
        return res


class ColIterableDF(Iterable[DataFrame]):
    __inner: TableBackend

    def __init__(self, backend: TableBackend):
        self.__inner = backend

    def __iter__(self) -> Iterator[DataFrame]:
        return ColIteratorDF(backend=self.__inner)