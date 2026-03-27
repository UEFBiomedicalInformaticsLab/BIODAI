from typing import Sequence

import pandas as pd
from pandas import DataFrame


def create_from_labelled_lists(lists: Sequence[list]) -> DataFrame:
    """Fills with NaN"""
    dfs = []
    for li in lists:
        dfs.append(DataFrame([li]))
    return pd.concat(dfs, )
