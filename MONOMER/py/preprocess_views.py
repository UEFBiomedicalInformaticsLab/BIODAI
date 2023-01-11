import pandas as pd

from util.printer.printer import Printer


def generic_preprocess_one_view(view, printer: Printer):
    res = view.copy()
    columns = list(res)
    for i in columns:
        if pd.api.types.is_numeric_dtype(res[i]):
            res[i] = res[i].astype(float)
    if res.dtypes.nunique() > 1:
        printer.print("Mixed type!")
        printer.print(res.dtypes.unique())
        printer.print(res)
    printer.print("Types after preprocessing: " + str(res.dtypes.unique()))
    return res


def preprocess_views(views, printer: Printer):
    res = {}
    for v in views:
        printer.print("Preprocessing view " + v)
        res[v] = generic_preprocess_one_view(views[v], printer=printer)
    return res
