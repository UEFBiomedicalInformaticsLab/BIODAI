from pandas import DataFrame

from feature_importance.feature_importance import FeatureImportance
from util.distribution.uniform_distribution import UniformDistribution
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table
from util.table.table_utils import n_col


class FeatureImportanceUniform(FeatureImportance):

    def compute(self, x: Table, y=None, n_proc: int = 1,
                printer: Printer = UNBUFFERED_OUT_PRINTER) -> UniformDistribution:
        return UniformDistribution(size=n_col(x))

    def compute_df(self, x: DataFrame, y=None, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> UniformDistribution:
        return UniformDistribution(size=n_col(x))  # Works for pandas and numpy and tables

    def nick(self) -> str:
        return "uniformFI"

    def name(self) -> str:
        return "uniform FI"

    def __str__(self) -> str:
        return "uniform feature importance"
