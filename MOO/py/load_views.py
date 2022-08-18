import pandas as pd
import os

from util.printer.printer import Printer


def load_view(dir, file):
    to_load_path = os.path.join(dir, file)
    dat = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")
    return dat


def load_view_from_type(dir, view_type):
    return load_view(dir, view_type + ".csv")


def load_all_views(dir, view_types, printer: Printer):
    res = {}
    for v in view_types:
        printer.print("Loading view " + v)
        v_res = load_view_from_type(dir, v)
        res[v] = v_res
    return res
