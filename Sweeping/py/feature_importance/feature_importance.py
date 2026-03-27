from abc import abstractmethod

from pandas import DataFrame

from input_data.outcome import Outcome
from util.distribution.distribution import Distribution
from util.named import NickNamed
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table


class FeatureImportance(NickNamed):

    @abstractmethod
    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        """Returned list assigns an importance to each feature. x should be a single view."""
        raise NotImplementedError()

    @abstractmethod
    def compute_df(
            self, x: DataFrame, y: DataFrame, n_proc: int = 1,
            printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        """Returned list assigns an importance to each feature. x should be a single view."""
        raise NotImplementedError()
