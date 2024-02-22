import load_views
import numpy as np

from util.printer.printer import Printer

outcome_name = "outcome"
MIRNA_NAME = "mirna"
MRNA_NAME = "mrna"
LOG_MRNA_NAME = "log_mrna"  # mrna already log transformed.
RPPAA_NAME = "rppaa"
METH_NAME = "meth"
AGE_NAME = "age"
view_types = ["cnasnp", METH_NAME, MIRNA_NAME, MRNA_NAME, LOG_MRNA_NAME, "mutations", RPPAA_NAME]
view_types_small = [MIRNA_NAME, MRNA_NAME, RPPAA_NAME]


def transform_rna(view):
    """Applies log(x+1)"""
    res = np.log2(view + 1)
    return res


def load_all_views(directory, views, printer: Printer):
    res = load_views.load_all_views(directory, views, printer=printer)
    for v in res:
        if v == MIRNA_NAME or v == MRNA_NAME:
            printer.print("Applying transformation to view " + v)
            res[v] = transform_rna(res[v])
    return res


def load_outcome(directory):
    return load_views.load_view_from_type(directory, outcome_name)
