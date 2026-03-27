import os.path
from abc import abstractmethod, ABC

from util.named import NickNamed
from util.table.only_increasing_table_backend.only_increasing_backend_with_retry import \
    OnlyIncreasingTableBackendWithRetry
from util.table.table import Table
import util.table.only_increasing_table_backend.hdf5_table as hdf5_table
from util.table.only_increasing_table_backend.hdf5_chunk_strategy import TRANSPOSE_HDF5
from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingAdapter
from util.table.transposed_table import TransposedTable


class TableRepresentationStrategy(NickNamed, ABC):

    @abstractmethod
    def represent(self, table: Table, directory: str, table_name: str) -> Table:
        raise NotImplementedError()


class TableRepresentationStrategyMemory(TableRepresentationStrategy):

    def represent(self, table: Table, directory: str, table_name: str) -> Table:
        return table.compile(max_cells=None)

    def nick(self) -> str:
        return "memory"


class TableRepresentationStrategyDisk(TableRepresentationStrategy):

    def represent(self, table: Table, directory: str, table_name: str) -> Table:
        if table.has_fast_cols():
            return table
        else:
            if TRANSPOSE_HDF5:
                table = TransposedTable(table=table)
            from util.table.backed_table import BackedTable
            res = BackedTable(backend=OnlyIncreasingAdapter(
                OnlyIncreasingTableBackendWithRetry(
                    inner=hdf5_table.HDF5Table.create_file(
                        filename=os.path.join(directory, table_name + ".hdf5"), data=table))))
            if TRANSPOSE_HDF5:
                res = TransposedTable(res)
        return res

    def nick(self) -> str:
        return "disk"
