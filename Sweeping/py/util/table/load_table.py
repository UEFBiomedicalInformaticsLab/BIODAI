import os

from util.table.only_increasing_table_backend.csv_table import CSVTable
from util.table.only_increasing_table_backend.hdf5_chunk_strategy import TRANSPOSE_HDF5
from util.table.only_increasing_table_backend.hdf5_table import HDF5Table
from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingAdapter
from util.table.only_increasing_table_backend.only_increasing_backend_with_retry import \
    OnlyIncreasingTableBackendWithRetry
from util.table.table import Table
from util.table.transposed_table import TransposedTable


def load_table_by_path(path: str) -> Table:
    """Keeps row names. Handles transpose setting for hdf5."""
    from util.table.backed_table import BackedTable
    if os.path.isfile(path):
        if path.endswith(".csv"):
            table = BackedTable(backend=OnlyIncreasingAdapter(
                inner=OnlyIncreasingTableBackendWithRetry(inner=CSVTable(file_path=path))))
        elif path.endswith(".hdf5"):
            table = BackedTable(backend=OnlyIncreasingAdapter(
                inner=OnlyIncreasingTableBackendWithRetry(inner=HDF5Table(filename=path))))
            if TRANSPOSE_HDF5:
                table = TransposedTable(table)
        else:
            raise ValueError("Unsupported file type.")
        return table
    else:
        raise FileNotFoundError()


def load_table(directory: str, table_name: str) -> Table:
    """Keeps row names. Handles transpose setting for hdf5."""
    to_load_path_csv = os.path.join(directory, table_name + ".csv")
    to_load_path_hdf5 = os.path.join(directory, table_name + ".hdf5")
    if os.path.isfile(to_load_path_csv):
        return load_table_by_path(path=to_load_path_csv)
    elif os.path.isfile(to_load_path_hdf5):
        return load_table_by_path(path=to_load_path_hdf5)
    else:
        raise FileNotFoundError()
