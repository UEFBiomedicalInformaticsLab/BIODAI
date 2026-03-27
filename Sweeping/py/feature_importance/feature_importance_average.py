from collections.abc import Sequence

from pandas import DataFrame

from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome
from util.distribution.average_distribution import AverageDistribution
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.str_utils import str_in_lines
from util.table.table import Table


class FeatureImportanceAverage(FeatureImportance):
    __components: Sequence[FeatureImportance]

    def __init__(self, components: Sequence[FeatureImportance]):
        self.__components = components

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        return AverageDistribution([c.compute(x, y, n_proc=n_proc, printer=printer) for c in self.__components])

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        return AverageDistribution([c.compute_df(x, y, n_proc=n_proc, printer=printer) for c in self.__components])

    def nick(self) -> str:
        res = ""
        for c in self.__components:
            if res != "":
                res += "+"
            res += c.nick()
        return res

    def name(self) -> str:
        res = ""
        for c in self.__components:
            if res != "":
                res += " + "
            res += c.name()
        return res

    def __str__(self) -> str:
        res = "Average of:\n"
        res += str_in_lines(self.__components)
        return res
