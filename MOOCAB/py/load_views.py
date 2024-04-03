import pandas as pd
import os

from pandas import DataFrame

from util.dataframes import n_row, n_col
from util.printer.printer import Printer


def load_view(dir, file: str) -> DataFrame:
    """Keeps row names."""
    to_load_path = os.path.join(dir, file)
    dat = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")
    return dat


def load_view_from_type(dir, view_type: str) -> DataFrame:
    """Keeps row names."""
    return load_view(dir, view_type + ".csv")


def view_exists(dir, view_type: str) -> bool:
    to_load_path = os.path.join(dir, view_type + ".csv")
    return os.path.isfile(to_load_path)


def load_all_views(directory, view_types, printer: Printer) -> dict[str, DataFrame]:
    """Does not include checks for consistency between the views.
    Keeps row names."""
    res = {}
    for v in view_types:
        printer.print("Loading view " + v)
        if view_exists(directory, v):
            v_res = load_view_from_type(directory, v)
            res[v] = v_res
            printer.print("View loaded with " + str(n_row(v_res)) + " rows and " + str(n_col(v_res)) + " columns.")
        else:
            printer.print("View does not exist.")
    return res
