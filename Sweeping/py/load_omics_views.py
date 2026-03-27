import os
from collections.abc import Iterable

import pandas as pd
from pandas import DataFrame

import load_views
import numpy as np

from input_data.fallback_locations import FallbackLocations, EMPTY_FALLBACK_LOCATIONS
from util.dataframe.dataframes import has_negatives
from util.printer.printer import Printer
from util.table.backed_table import BackedTable
from util.table.table_backend.np_table import NpTable
from views.views import Views, JustViews

OUTCOME_NAME = "outcome"
MIRNA_NAME = "mirna"
LOG_MIRNA_NAME = "log_mirna"  # mirna already log transformed.
MRNA_NAME = "mrna"
LOG_MRNA_NAME = "log_mrna"  # mrna already log transformed.
RPPAA_NAME = "rppaa"
METH_NAME = "meth"
AGE_NAME = "age"
CLINIC_NAME = "clinic"
SNP_NAME = "snp"
PROTEOMICS_NAME = "proteomics"
VIEW_TYPES_SMALL = [MIRNA_NAME, MRNA_NAME, RPPAA_NAME]


def transform_rna(view: DataFrame) -> DataFrame:
    """Applies log(x+1)"""
    res = np.log2(view + 1.0)
    return res


def load_all_views(directory, views: Iterable[str], printer: Printer,
                   fallback_locations: FallbackLocations = EMPTY_FALLBACK_LOCATIONS) -> Views:
    """Does not include checks for consistency between the views. Keeps row names.
    Applies log transformation to MRNA and MiRNA."""
    loaded = load_views.load_all_views(directory, views, printer=printer, fallback_locations=fallback_locations)
    res = {}
    for v in loaded.keys():
        res[v] = loaded.view(v)
        if v == MIRNA_NAME or v == MRNA_NAME:
            printer.print("Applying transformation to view " + v)
            df = res[v].to_dataframe()
            if has_negatives(df=df):
                printer.print("Warning: negative values before logarithm.")
            res[v] = BackedTable(NpTable(data=transform_rna(df)))
    return JustViews(views_dict=res)


def load_outcome_by_path(path: str) -> DataFrame:
    """Keeps row names."""
    return pd.read_csv(path, index_col=0)


def load_outcome(directory) -> DataFrame:
    """Keeps row names. We cannot load a Table because tables do not accept categorical outcomes if they are encoded
    with strings."""
    file_name = os.path.join(directory, OUTCOME_NAME + ".csv")
    return load_outcome_by_path(path=file_name)
