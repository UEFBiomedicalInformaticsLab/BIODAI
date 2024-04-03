from pandas import DataFrame

import load_views
import numpy as np

from util.dataframes import has_negatives
from util.printer.printer import Printer

OUTCOME_NAME = "outcome"
MIRNA_NAME = "mirna"
MRNA_NAME = "mrna"
LOG_MRNA_NAME = "log_mrna"  # mrna already log transformed.
RPPAA_NAME = "rppaa"
METH_NAME = "meth"
AGE_NAME = "age"
CLINIC_NAME = "clinic"
VIEW_TYPES_SMALL = [MIRNA_NAME, MRNA_NAME, RPPAA_NAME]


def transform_rna(view: DataFrame) -> DataFrame:
    """Applies log(x+1)"""
    res = np.log2(view + 1.0)
    return res


def load_all_views(directory, views, printer: Printer) -> dict[str, DataFrame]:
    """Does not include checks for consistency between the views. Keeps row names."""
    res = load_views.load_all_views(directory, views, printer=printer)
    for v in res:
        if v == MIRNA_NAME or v == MRNA_NAME:
            printer.print("Applying transformation to view " + v)
            res_v = res[v]
            if has_negatives(df=res_v):
                printer.print("Warning: negative values before logarithm.")
            res[v] = transform_rna(res_v)
    return res


def load_outcome(directory) -> DataFrame:
    """Keeps row names."""
    return load_views.load_view_from_type(directory, OUTCOME_NAME)
