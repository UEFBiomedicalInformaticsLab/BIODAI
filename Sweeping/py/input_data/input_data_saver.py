import os

from pandas import DataFrame

from input_data.input_data import InputData
from util.dataframe.dataframes import cbind
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.table.table_representation_strategy import TableRepresentationStrategyDisk
from util.table.table_to_csv import table_to_csv


MAX_CSV_CELLS = 50000000


def smart_save_table_unchecked(directory: str, table_name: str, table: Table):
    """Does not check if there is a table with the same name and overrides if needed."""
    n_cells = table.n_cells()
    view_path_csv = os.path.join(directory, table_name + ".csv")
    if n_cells <= MAX_CSV_CELLS:
        table_to_csv(table=table, file_path=view_path_csv, overwrite=True)
    else:
        TableRepresentationStrategyDisk().represent(table=table, directory=directory, table_name=table_name)


def smart_save_df_unchecked(directory: str, table_name: str, df: DataFrame):
    """Does not check if there is a table with the same name and overrides if needed."""
    table = BackedTable(backend=NpTable(data=df))
    smart_save_table_unchecked(directory=directory, table_name=table_name, table=table)


def save_input_view(directory: str, view_name: str, view_table: Table):
    view_path_csv = os.path.join(directory, view_name + ".csv")
    view_path_hdf5 = os.path.join(directory, view_name + ".hdf5")
    if not os.path.isfile(view_path_csv) and not os.path.isfile(view_path_hdf5):
        smart_save_table_unchecked(directory=directory, table_name=view_name, table=view_table)


def save_input_data(data: InputData, directory: str):
    for view_name in data.view_names_seq():
        save_input_view(directory=directory, view_name=view_name, view_table=data.view(view_name=view_name))
    outcome_path = os.path.join(directory, "outcome" + ".csv")
    if not os.path.isfile(outcome_path):
        outcome_datas = [o.data() for o in data.outcomes()]
        outcome_df = cbind(outcome_datas)
        outcome_df.to_csv(outcome_path, index=True, mode='w')
