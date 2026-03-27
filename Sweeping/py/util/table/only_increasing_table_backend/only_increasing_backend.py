from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Sequence, Optional

from numpy import ndarray, ix_
from pandas import DataFrame, Index
from sortedcontainers import SortedSet

from util.sequence_utils import strictly_increasing
from util.table.backed_table import BackedTable
from util.table.table import ChunkIterable, Table, ChunkIterableDF
from util.table.table_backend import np_table
from util.table.table_backend.table_backend import TableBackend


class OnlyIncreasingChunkIterable(ChunkIterable):

    def __init__(self, backend: OnlyIncreasingTableBackend, chunk_rows: Optional[int] = None,
                 selected_rows: Optional[Sequence[int]] = None, selected_cols: Optional[Sequence[int]] = None):
        """Selected rows and columns do not need to be sorted since the selection is applied
        after creating a Table."""
        ChunkIterable.__init__(
            self=self, table=BackedTable(
            backend=OnlyIncreasingAdapter(backend),
            selected_rows=selected_rows, selected_cols=selected_cols),
            chunk_rows=chunk_rows)


class OnlyIncreasingChunkIterableDF(ChunkIterableDF):

    def __init__(self, backend: OnlyIncreasingTableBackend, chunk_rows: Optional[int] = None,
                 selected_rows: Optional[Sequence[int]] = None, selected_cols: Optional[Sequence[int]] = None):
        """In this implementation, selected rows and columns do not need to be sorted since the selection is applied
        after creating a Table."""
        ChunkIterableDF.__init__(
            self=self, table=BackedTable(
            backend=OnlyIncreasingAdapter(backend),
            selected_rows=selected_rows, selected_cols=selected_cols),
            chunk_rows=chunk_rows)


class OnlyIncreasingTableBackend(ABC):
    """Similar to a TableBackend, but accepts only increasing indices without repetitions."""

    @abstractmethod
    def n_row(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def n_col(self) -> int:
        raise NotImplementedError()

    def size(self) -> int:
        return self.n_row() * self.n_col()

    @abstractmethod
    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        """Returned object is either new or immutable."""
        raise NotImplementedError()

    def to_dataframe(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> DataFrame:
        """Returned object is either new or immutable."""
        res_np = self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
        res_rownames = self.rownames(selected=selected_rows)
        res_colnames = self.colnames(selected=selected_cols)
        return DataFrame(data=res_np, index=Index(res_rownames), columns=Index(res_colnames))

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

    def chunk_iterable(self, chunk_rows: Optional[int],
                       selected_rows: Optional[Sequence[int]],
                       selected_cols: Optional[Sequence[int]]) -> Iterable[Table]:
        """Selected rows and columns do not need to be sorted and can have repetitions."""
        return OnlyIncreasingChunkIterable(backend=self, chunk_rows=chunk_rows,
                                           selected_rows=selected_rows, selected_cols=selected_cols)

    def chunk_iterable_df(self, chunk_rows: Optional[int],
                       selected_rows: Optional[Sequence[int]],
                       selected_cols: Optional[Sequence[int]]) -> Iterable[DataFrame]:
        """Indices must be increasing and without repetitions. This allows for faster specializations."""
        return OnlyIncreasingChunkIterableDF(backend=self, chunk_rows=chunk_rows,
                                             selected_rows=selected_rows, selected_cols=selected_cols)

    @abstractmethod
    def has_fast_cols(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_fast_rows(self) -> bool:
        raise NotImplementedError()


class OnlyIncreasingAdapter(TableBackend):
    __inner: OnlyIncreasingTableBackend

    def __init__(self, inner: OnlyIncreasingTableBackend):
        self.__inner = inner

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    @staticmethod
    def __to_extract_rows_cols(
            selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> tuple[
        Optional[SortedSet[int]], Optional[dict[int, int]], Optional[SortedSet[int]], Optional[dict[int, int]]]:
        index_map_rows = None
        index_map_cols = None
        to_extract_rows = None
        to_extract_cols = None
        if selected_rows is not None:
            to_extract_rows, index_map_rows = OnlyIncreasingAdapter.__to_extract_and_map(selected=selected_rows)
        if selected_cols is not None:
            to_extract_cols, index_map_cols = OnlyIncreasingAdapter.__to_extract_and_map(selected=selected_cols)
        return to_extract_rows, index_map_rows, to_extract_cols, index_map_cols

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        to_extract_rows, index_map_rows, to_extract_cols, index_map_cols = self.__to_extract_rows_cols(
            selected_rows=selected_rows, selected_cols=selected_cols)
        extracted = self.__inner.to_numpy(selected_rows=to_extract_rows, selected_cols=to_extract_cols)
        if selected_rows is None:
            if selected_cols is None:
                return extracted
            else:
                return extracted[:, [index_map_cols[s] for s in selected_cols]]
        else:
            if selected_cols is None:
                return extracted[[index_map_rows[s] for s in selected_rows], :]
            else:
                return extracted[
                    ix_([index_map_rows[s] for s in selected_rows], [index_map_cols[s] for s in selected_cols])]

    def __numpy_and_labels(
            self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]
    ) -> tuple[ndarray, Sequence[str], Sequence[str]]:
        """Code is a bit convoluted to allow for reuse of the index maps that can be expensive to build."""
        to_extract_rows, index_map_rows, to_extract_cols, index_map_cols = self.__to_extract_rows_cols(
            selected_rows=selected_rows, selected_cols=selected_cols)
        extracted = self.__inner.to_numpy(selected_rows=to_extract_rows, selected_cols=to_extract_cols)
        extracted_rownames = self.__inner.rownames(selected=to_extract_rows)
        extracted_colnames = self.__inner.colnames(selected=to_extract_cols)
        if selected_rows is None:
            if selected_cols is None:
                res_np = extracted
                res_rownames = extracted_rownames
                res_colnames = extracted_colnames
            else:
                res_np = extracted[:, [index_map_cols[s] for s in selected_cols]]
                res_rownames = extracted_rownames
                res_colnames = [extracted_colnames[index_map_cols[s]] for s in selected_cols]
        else:
            if selected_cols is None:
                res_np = extracted[[index_map_rows[s] for s in selected_rows], :]
                res_rownames = [extracted_rownames[index_map_rows[s]] for s in selected_rows]
                res_colnames = extracted_colnames
            else:
                res_np = extracted[
                    ix_([index_map_rows[s] for s in selected_rows], [index_map_cols[s] for s in selected_cols])]
                res_rownames = [extracted_rownames[index_map_rows[s]] for s in selected_rows]
                res_colnames = [extracted_colnames[index_map_cols[s]] for s in selected_cols]
        return res_np, res_rownames, res_colnames

    def to_dataframe(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> DataFrame:
        to_extract_rows, index_map_rows, to_extract_cols, index_map_cols = self.__to_extract_rows_cols(
            selected_rows=selected_rows, selected_cols=selected_cols)
        extracted = self.__inner.to_dataframe(selected_rows=to_extract_rows, selected_cols=to_extract_cols)
        if selected_rows is None:
            if selected_cols is None:
                return extracted
            else:
                return extracted.iloc[:, [index_map_cols[s] for s in selected_cols]]
        else:
            if selected_cols is None:
                return extracted.iloc[[index_map_rows[s] for s in selected_rows], :]
            else:
                selected_rows_indices = [index_map_rows[s] for s in selected_rows]
                selected_cols_indices = [index_map_cols[s] for s in selected_cols]
                return extracted.iloc[selected_rows_indices, selected_cols_indices]

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]
                ) -> TableBackend:
        res_np, res_rownames, res_colnames = self.__numpy_and_labels(
            selected_rows=selected_rows, selected_cols=selected_cols)
        return np_table.NpTable(data=res_np, rownames=res_rownames, colnames=res_colnames)

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__inner.colnames(selected=None)
        else:
            to_extract, index_map = self.__to_extract_and_map(selected=selected)
            extracted = self.__inner.colnames(selected=to_extract)
            return [extracted[index_map[s]] for s in selected]

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            return self.__inner.rownames(selected=None)
        else:
            to_extract, index_map = self.__to_extract_and_map(selected=selected)
            extracted = self.__inner.rownames(selected=to_extract)
            return [extracted[index_map[s]] for s in selected]

    @staticmethod
    def __to_extract_and_map(selected: Sequence[int]) -> tuple[SortedSet[int], dict[int, int]]:
        to_extract = SortedSet(selected)
        index_map = {}
        for i, e in enumerate(to_extract):
            index_map[e] = i
        return to_extract, index_map

    def chunk_iterable_df(self, chunk_rows: Optional[int],
                          selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]
                          ) -> Iterable[DataFrame]:
        if (selected_rows is None or strictly_increasing(selected_rows)) and (selected_cols is None or strictly_increasing(selected_cols)):
            """This may allow for faster specializations."""
            return self.__inner.chunk_iterable_df(chunk_rows=chunk_rows, selected_rows=selected_rows, selected_cols=selected_cols)
        else:
            return TableBackend.chunk_iterable_df(
                self=self, chunk_rows=chunk_rows, selected_rows=selected_rows, selected_cols=selected_cols)

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()