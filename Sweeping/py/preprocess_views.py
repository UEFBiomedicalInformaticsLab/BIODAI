import pandas as pd
from pandas import DataFrame

from util.printer.printer import Printer
from util.table.backed_table import BackedTable
from util.table.table_backend.np_table import NpTable
from views.views import Views, JustViews


def generic_preprocess_one_view(view: DataFrame, printer: Printer) -> DataFrame:
    """Legacy code. It was used when views were not loaded directly as tables."""
    res = view.copy()
    columns = list(res)
    for i in columns:
        res_i = res[i]
        if not pd.api.types.is_float_dtype(res_i):  # If it is already ok no need for further computation.
            if pd.api.types.is_numeric_dtype(res_i):
                res[i] = res_i.astype(float)
    if res.dtypes.nunique() > 1:
        printer.print("Mixed type!")
        printer.print(res.dtypes.unique())
        printer.print(res)
    printer.print("Types after preprocessing: " + str(res.dtypes.unique()))
    return res


def preprocess_views(views: Views, printer: Printer) -> Views:
    """Legacy code. It was used when views were not loaded directly as tables."""
    res = {}
    for v in views.keys():
        printer.print("Preprocessing view " + v)
        res[v] = BackedTable(NpTable(generic_preprocess_one_view(view=views.view(v).to_dataframe(), printer=printer)))
    return JustViews(res)
