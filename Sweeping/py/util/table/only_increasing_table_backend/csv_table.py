import csv
from _csv import Error
from collections.abc import Iterable, Iterator
from typing import Optional, Sequence

import pandas as pd
from frozenlist import FrozenList
from numpy import ndarray
from pandas import DataFrame

from util.math.utils import to_np_float64, to_float
from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingTableBackend
from util.table.table import clean_chunk_rows
from util.table.table_utils import n_row


INDEX_COL = 0


def to_float_factory():
    return to_np_float64


def sniff_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as file:
        sample = file.read(1024)
        sniffer = csv.Sniffer()
        try:
            res = str(sniffer.sniff(sample).delimiter)
        except Error as e:
            print("Not able to sniff the delimiter for " + str(file_path))
            print("Will try to proceed with default comma delimiter.")
            res = ","
        return res


class CSVTable(OnlyIncreasingTableBackend):
    """Rownames and colnames are cached after first read for speed.
    Reads lazily on demand. All values are converted to float, NaN when not possible."""
    __file_path: str
    __chunk_rows: int
    __has_index_col: bool
    __rownames: Optional[FrozenList[str]]
    __colnames: Optional[FrozenList[str]]
    __delimiter: Optional[str]

    def __init__(self, file_path: str, chunk_rows: Optional[int] = None,
                 has_index_col: bool = True, delimiter: Optional[str] = None):
        self.__file_path = file_path
        self.__has_index_col = has_index_col
        self.__rownames = None
        self.__colnames = None
        self.__delimiter = delimiter
        if chunk_rows is None:
            self.__chunk_rows = clean_chunk_rows(num_col=self.n_col())

    def n_row(self) -> int:
        return len(self.rownames(selected=None))

    def n_col(self) -> int:
        res = len(self.colnames(selected=None))
        return res

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        return self.to_dataframe(selected_rows=selected_rows, selected_cols=selected_cols).to_numpy(copy=False)

    def memory_size(self) -> int:
        return 0

    def __index_col(self) -> Optional[int]:
        if self.__has_index_col:
            return 0
        else:
            return None

    def __sep(self) -> str:
        if self.__delimiter is None:
            self.__delimiter = sniff_delimiter(file_path=self.__file_path)
        return self.__delimiter

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if self.__colnames is None:
            cols = pd.read_csv(filepath_or_buffer=self.__file_path,
                        index_col=self.__index_col(),
                        sep=self.__sep(),
                        nrows=0).columns.tolist()
            self.__colnames = FrozenList(items=cols)
            self.__colnames.freeze()
        if selected is None:
            return self.__colnames
        else:
            return [self.__colnames[s] for s in selected]

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if self.__rownames is None:
            df = pd.read_csv(filepath_or_buffer=self.__file_path, usecols=[0], index_col=0,
                             sep=self.__sep())
            self.__rownames = FrozenList(items=[str(s) for s in df.index])
            self.__rownames.freeze()
        if selected is None:
            return self.__rownames
        else:
            return [self.__rownames[s] for s in selected]

    def pd_reader(self, selected_cols: Optional[Sequence[int]]):
        if selected_cols is None:
            selected_cols = range(0, self.n_col())
        if len(selected_cols) == 0:
            raise ValueError("Pandas read_csv returns [] when there are no selected columns.")
        # Maybe it is an unintended behaviour, but it seems that the converters must be indexed according to
        # the positions after the selection of the columns.
        if self.__has_index_col:
            selected_cols = [INDEX_COL] + [c+1 for c in selected_cols]  # To skip the index col
            converters = {col: to_float for col in range(1,len(selected_cols))}
            converters[INDEX_COL] = str
        else:
            converters = {col: to_float for col in range(len(selected_cols))}
        index_col = self.__index_col()
        return pd.read_csv(filepath_or_buffer=self.__file_path,
                           index_col=index_col, usecols=selected_cols,
                           chunksize=clean_chunk_rows(chunk_rows=self.__chunk_rows, num_col=self.n_col()),
                           converters=converters, sep=self.__sep(), low_memory=True)

    def to_dataframe(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> DataFrame:
        reader = self.pd_reader(selected_cols=selected_cols)
        to_concat = []
        save_rownames = self.__rownames is None
        collected_rownames = []
        current_index = 0
        for chunk in reader:
            chunk_rows = n_row(chunk)
            chunk.index = chunk.index.astype(str)
            # Because it can happen for some reason that the index is not read as strings as expected.
            if save_rownames:
                collected_rownames.extend(chunk.index)
            if selected_rows is None:
                to_concat.append(chunk)
            else:
                to_concat.append(
                    chunk.iloc[[i - current_index for i in selected_rows if current_index <= i < current_index + chunk_rows]])
            current_index += chunk_rows
        if save_rownames:
            self.__rownames = FrozenList(items=collected_rownames)
            self.__rownames.freeze()
        res = pd.concat(to_concat)
        # res.columns = self.colnames(selected=selected_cols)
        return res

    def chunk_iterable_df(
            self, chunk_rows: Optional[int],
            selected_rows: Optional[Sequence[int]],
            selected_cols: Optional[Sequence[int]]) -> Iterable[DataFrame]:
        """Indices must be increasing and without repetitions."""
        return CSVChunkIterableDF(
            csv_table=self,
            chunk_rows=clean_chunk_rows(num_col=self.n_col(), chunk_rows=chunk_rows),
            selected_rows=selected_rows, selected_cols=selected_cols)

    def has_fast_cols(self) -> bool:
        return False

    def has_fast_rows(self) -> bool:
        return False


class CSVChunkIteratorDF(Iterator[DataFrame]):
    __reader: any
    __chunk_rows: int
    __selected_rows: Optional[Sequence[int]]
    __leftover: Optional[DataFrame]


    def __init__(self, reader, chunk_rows: int, selected_rows: Optional[Sequence[int]]):
        if chunk_rows < 1:
            raise ValueError()
        self.__chunk_rows = chunk_rows
        self.__reader = reader
        self.__selected_rows = selected_rows
        self.__leftover = None

    def __next__(self) -> DataFrame:
        to_concat = []
        if self.__leftover is not None:
            to_concat.append(self.__leftover)
            self.__leftover = None
        proceed = True
        while sum(n_row(tc) for tc in to_concat) < self.__chunk_rows and proceed:
            try:
                chunk = next(self.__reader)
                if self.__selected_rows is None:
                    to_concat.append(chunk)
                else:
                    to_concat.append(chunk.iloc[self.__selected_rows])
            except StopIteration:
                proceed = False
        if len(to_concat) == 0:
            raise StopIteration()
        all_rows = pd.concat(to_concat)
        n_all = n_row(all_rows)
        if n_all == 0:
            raise StopIteration()
        if n_all > self.__chunk_rows:
            res = all_rows.iloc[:self.__chunk_rows,:]
            self.__leftover = all_rows.iloc[self.__chunk_rows:,:]
            return res
        else:
            return all_rows


class CSVChunkIterableDF(Iterable[DataFrame]):
    __inner: CSVTable
    __chunk_rows: int
    __selected_rows: Optional[Sequence[int]]
    __selected_cols: Optional[Sequence[int]]

    def __init__(self,
                 csv_table: CSVTable,
                 chunk_rows: int,
                 selected_rows: Optional[Sequence[int]],
                 selected_cols: Optional[Sequence[int]],
                 ):
        if chunk_rows < 1:
            raise ValueError()
        self.__inner=csv_table
        self.__chunk_rows=chunk_rows
        self.__selected_rows=selected_rows
        self.__selected_cols=selected_cols

    def __iter__(self) -> Iterator[DataFrame]:
        reader = self.__inner.pd_reader(selected_cols=self.__selected_cols)
        return CSVChunkIteratorDF(reader=reader, chunk_rows=self.__chunk_rows, selected_rows=self.__selected_rows)
