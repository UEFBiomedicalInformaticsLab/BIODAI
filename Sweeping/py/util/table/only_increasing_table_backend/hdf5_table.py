from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Sequence, Optional, Any

import h5py
from frozenlist import FrozenList
from numpy import ndarray
from pandas import DataFrame
from static_frame import Frame

from util.progress_observer import ProgressObserverFactory, DEFAULT_PROGRESS_OBSERVER_FACTORY
from util.table.only_increasing_table_backend.hdf5_chunk_strategy import HDF5ChunkStrategy, DEFAULT_HDF5_CHUNK_STRATEGY, \
    good_indexing
from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingTableBackend
from util.table.table import Table, clean_chunk_rows
from util.table.table_backend.np_table import NpTable
from util.table.table_backend.table_backend import TableBackend
from util.table.table_utils import n_row, n_col, is_table_data, is_2d

HDF5_DATASET_NAME = "dataset"
HDF5_ROWNAMES = "rownames"
HDF5_COLNAMES = "colnames"


class HDF5FileHandler(ABC):

    @abstractmethod
    def __enter__(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError()


class StatelessHDF5FileHandler(HDF5FileHandler):
    __filename: str
    __file: Any

    def __init__(self, filename: str):
        self.__filename = filename
        self.__file = None

    def __enter__(self) -> Any:
        self.__file = h5py.File(name=self.__filename, mode='r', rdcc_w0=0, rdcc_nbytes=0, rdcc_nslots=0)
        return self.__file

    def __exit__(self, type, value, traceback):
        if self.__file is not None:
            self.__file.close()
            self.__file = None


class CachedHDF5FileHandler(HDF5FileHandler):
    __filename: str
    __file: Any
    __verbose: bool
    __chunk_strategy: HDF5ChunkStrategy

    def __init__(self, filename: str, chunk_strategy: HDF5ChunkStrategy, verbose: bool = False):
        self.__filename = filename
        self.__file = None
        self.__verbose = verbose
        self.__chunk_strategy = chunk_strategy

    def __enter__(self) -> Any:
        if self.__file is None:
            with h5py.File(name=self.__filename, mode='r', rdcc_w0=0, rdcc_nbytes=0, rdcc_nslots=0) as f:
                nrow = f[HDF5_DATASET_NAME].shape[0]
                ncol = f[HDF5_DATASET_NAME].shape[1]
            rdcc_nbytes = self.__chunk_strategy.optimal_rdcc_nbytes(nrow=nrow, ncol=ncol)
            rdcc_nslots = self.__chunk_strategy.optimal_rdcc_nslots(nrow=nrow, ncol=ncol)
            if self.__verbose:
                print("OPENING HDF5 FILE\n" +
                      "nrow: " + str(nrow) + "\n" +
                      "ncol: " + str(ncol) + "\n" +
                      "rdcc_nbytes: " + str(rdcc_nbytes) + "\n" +
                      "rdcc_nslots: " + str(rdcc_nslots) + "\n")
            self.__file = h5py.File(name=self.__filename, mode='r', rdcc_w0=0,
                                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        return self.__file

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        if self.__file is not None:
            to_close = self.__file
            self.__file = None  # Reduce probability of closing two times concurrently.
            if self.__verbose:
                print("CLOSING HDF5 FILE\n")
            try:
                to_close.close()
            except TypeError as e:
                print("Exception while closing hdf5 file, program will continue.")
                print("Exception:\n" + str(e))

    def __copy__(self):
        """We do not copy the cache."""
        return CachedHDF5FileHandler(
            filename=self.__filename, verbose=self.__verbose, chunk_strategy=self.__chunk_strategy)

    def __deepcopy__(self, memodict=None):
        """We do not copy the cache."""
        return CachedHDF5FileHandler(
            filename=self.__filename, verbose=self.__verbose, chunk_strategy=self.__chunk_strategy)

    def __getstate__(self):
        """We do not pickle the cache."""
        return self.__filename, self.__verbose, self.__chunk_strategy

    def __setstate__(self, state):
        self.__filename = state[0]
        self.__verbose = state[1]
        self.__chunk_strategy = state[2]
        self.__file = None

    def chunk_strategy(self) -> HDF5ChunkStrategy:
        return self.__chunk_strategy

    def has_fast_cols(self) -> bool:
        return self.__chunk_strategy.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__chunk_strategy.has_fast_rows()


class HDF5Table(OnlyIncreasingTableBackend):
    """Rownames and colnames are cached after first read for speed."""
    __file_handler: CachedHDF5FileHandler
    __rownames: Optional[FrozenList[str]]
    __colnames: Optional[FrozenList[str]]
    __n_row: Optional[int]
    __n_col: Optional[int]

    def __init__(self, filename: str,
                 chunk_strategy: HDF5ChunkStrategy = DEFAULT_HDF5_CHUNK_STRATEGY):
        self.__file_handler = CachedHDF5FileHandler(filename=filename, chunk_strategy=chunk_strategy)
        self.__rownames = None
        self.__colnames = None
        self.__n_row = None
        self.__n_col = None

    def n_row(self) -> int:
        if self.__n_row is None:
            with self.__file_handler as f:
                self.__n_row = f[HDF5_DATASET_NAME].shape[0]
        return self.__n_row

    def n_col(self) -> int:
        if self.__n_col is None:
            with self.__file_handler as f:
                self.__n_col = f[HDF5_DATASET_NAME].shape[1]
        return self.__n_col

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        """h5py does not support indices that are not ordered."""
        with self.__file_handler as f:
            data = f[HDF5_DATASET_NAME]
            if selected_rows is None:
                if selected_cols is None:
                    res = data[:, :]  # We need the [:, :] otherwise it is not a numpy array.
                else:
                    res = data[:, good_indexing(selected_cols)]
            else:
                if selected_cols is None:
                    res = data[good_indexing(selected_rows), :]
                else:
                    # Chunk strategy optimizes the order of the selects based on the chunk shape.
                    res = self.__file_handler.chunk_strategy().select(
                        data=data, selected_cols=selected_cols, selected_rows=selected_rows)
        if isinstance(res, ndarray):
            return res
        else:
            raise ValueError("Result is not a numpy array as expected.")

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        """h5py does not support indices that are not ordered."""
        if self.__colnames is None:
            with self.__file_handler as f:
                if HDF5_COLNAMES in f:
                    names = f[HDF5_COLNAMES].asstr()
                    self.__colnames = FrozenList(items=names[:])
                else:
                    self.__colnames = FrozenList(items=(str(i) for i in range(self.n_col())))
                self.__colnames.freeze()
        if selected is None:
            return self.__colnames
        else:
            return [self.__colnames[s] for s in selected]

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        """h5py does not support indices that are not ordered."""
        if self.__rownames is None:
            with self.__file_handler as f:
                if HDF5_ROWNAMES in f:
                    names = f[HDF5_ROWNAMES].asstr()
                    self.__rownames = FrozenList(items=names[:])
                else:
                    self.__rownames = FrozenList(items=(str(i) for i in range(self.n_row())))
                self.__rownames.freeze()
        if selected is None:
            return self.__rownames
        else:
            return [self.__rownames[s] for s in selected]

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        """h5py does not support indices that are not ordered."""
        return NpTable(
            data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
            rownames=self.rownames(selected=selected_rows), colnames=self.colnames(selected=selected_cols))

    @staticmethod
    def create_file(filename: str, data: Union[ndarray, DataFrame, Frame, Table],
                    chunk_strategy: HDF5ChunkStrategy = DEFAULT_HDF5_CHUNK_STRATEGY,
                    progress_fact: ProgressObserverFactory = DEFAULT_PROGRESS_OBSERVER_FACTORY) -> HDF5Table:
        """Does not create a new file if it exists. Creates also directories if needed.
        """
        if not is_2d(data=data):
            raise ValueError("Data should be 2D.")
        if not is_table_data(data=data):
            raise ValueError("Data should be numeric.")
        file_exist = os.path.isfile(filename)
        if file_exist:
            res = HDF5Table(filename=filename, chunk_strategy=chunk_strategy)
            if res.n_row() == n_row(data) and res.n_col() == n_col(data):
                # Check to avoid using a clearly outdated file.
                return res
            else:
                del res
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        progress = progress_fact.create_progress_observer(job_name="Creating new hdf5 file")
        progress.notify_start()
        progress.notify_message(text=str(filename))
        num_rows = n_row(data)
        num_cols = n_col(data)
        progress.notify_message(text="Must write " + str(num_rows) + " rows and " + str(num_cols) + " columns.")
        try:
            with h5py.File(name=filename, mode='w') as f:
                # Handle row and column names
                if isinstance(data, DataFrame) or isinstance(data, Frame) or isinstance(data, Table):
                    if isinstance(data, DataFrame) or isinstance(data, Frame):
                        rownames = [str(n) for n in data.index]
                        colnames = [str(n) for n in data.columns]
                    else:
                        rownames = list(data.rownames())
                        colnames = list(data.colnames())
                        # They are cast to lists since FrozenList can give problems with hdf5 writer.
                    f.create_dataset(
                        name=HDF5_ROWNAMES, data=rownames,
                        compression="gzip", chunks=(len(rownames),))
                    f.create_dataset(
                        name=HDF5_COLNAMES, data=colnames,
                        compression="gzip", chunks=(len(colnames),))

                # Handle data writing in chunks
                progress.notify_message(text="Writing in chunks.")
                if isinstance(data, Table):  # File is created in chunks to save memory.
                    data_shape = data.shape()
                    row = 0
                    dset = f.create_dataset(HDF5_DATASET_NAME, shape=data_shape,
                                            chunks=chunk_strategy.chunks(data=data),
                                            compression="gzip")
                    for chunk in data.chunks():
                        chunk_rows = n_row(chunk)
                        dset[row:row+chunk_rows, 0:num_cols] = chunk.to_numpy()
                        row += chunk_rows
                        progress.notify_progress(proportion=row/num_rows, text=str(row) + "/" + str(num_rows) + " rows")
                else:
                    # Manually chunk non-Table data
                    chunk_size = clean_chunk_rows(num_col=num_cols)

                    # Create HDF5 dataset
                    dset = f.create_dataset(HDF5_DATASET_NAME, shape=(num_rows, num_cols),
                                            chunks=chunk_strategy.chunks(data=data),
                                            compression="gzip")

                    # Write data in chunks
                    for row in range(0, num_rows, chunk_size):
                        end_row = min(row + chunk_size, num_rows)

                        if isinstance(data, (DataFrame, Frame)):
                            chunk = data.iloc[row:end_row, :].values  # .values avoids .to_numpy() and may not copy
                        elif isinstance(data, ndarray):
                            chunk = data[row:end_row, :]

                        dset[row:end_row, :] = chunk
                        progress.notify_progress(proportion=end_row / num_rows,
                                                 text=str(end_row) + "/" + str(num_rows) + " rows")
        except (OSError, ValueError) as e:
            print("Error while trying to write the hdf5 cache file. Perhaps file already open."
                  "Will try to proceed with existing file.")
            print("Filename: " + str(filename))
            print("Exception:\n" + str(e))
        progress.notify_end()
        return HDF5Table(filename=filename, chunk_strategy=chunk_strategy)

    @staticmethod
    def create_file_old(filename: str, data: Union[ndarray, DataFrame, Frame, Table],
                    chunk_strategy: HDF5ChunkStrategy = DEFAULT_HDF5_CHUNK_STRATEGY) -> HDF5Table:
        """Does not create a new file if it exists. Creates also directories if needed."""
        if not is_2d(data):
            raise ValueError("Data should be 2D.")
        file_exist = os.path.isfile(filename)
        if file_exist:
            res = HDF5Table(filename=filename, chunk_strategy=chunk_strategy)
            if res.n_row() == n_row(data) and res.n_col() == n_col(data):
                # Check to avoid using a clearly outdated file.
                return res
            else:
                del res
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(name=filename, mode='w') as f:
                if isinstance(data, Table):
                    data = data.to_dataframe()
                f.create_dataset(
                    name=HDF5_DATASET_NAME, data=data, compression="gzip", chunks=chunk_strategy.chunks(data=data))
                if isinstance(data, DataFrame) or isinstance(data, Frame):
                    f.create_dataset(
                        name=HDF5_ROWNAMES, data=[str(n) for n in data.index],
                        compression="gzip", chunks=(n_row(data),))
                    f.create_dataset(
                        name=HDF5_COLNAMES, data=[str(n) for n in data.columns],
                        compression="gzip", chunks=(n_col(data),))
        except (OSError, ValueError) as e:
            print("Error while trying to write the hdf5 cache file. Perhaps file already open."
                  "Will try to proceed with existing file.")
            print("Filename: " + str(filename))
            print("Exception:\n" + str(e))
        return HDF5Table(filename=filename, chunk_strategy=chunk_strategy)

    def memory_size(self) -> int:
        """In practice, it is not necessarily zero because there might be caching."""
        return 0

    def has_fast_cols(self) -> bool:
        return self.__file_handler.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__file_handler.has_fast_rows()
