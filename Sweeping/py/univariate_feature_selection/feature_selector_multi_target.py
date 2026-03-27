from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame

from input_data.outcome import Outcome
from univariate_feature_selection.many_feature_selector import ManyFeatureSelector
from util.math.list_math import list_or
from util.named import NickNamed
from util.printer.printer import Printer, NullPrinter
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable


class FeatureSelectorMO(NickNamed,ABC):

    @abstractmethod
    def selection_mask(self, x: Table, outcomes: Sequence[Outcome],
                          printer: Printer = NullPrinter(), n_proc: int = 1,
                       covariates: Optional[Table] = None) -> list[bool]:
        raise NotImplementedError()

    @abstractmethod
    def selection_mask_df(self, x: DataFrame, outcomes: Sequence[Outcome],
                          printer: Printer = NullPrinter(), n_proc: int = 1,
                          covariates: Optional[Table] = None) -> list[bool]:
        """Override to provide faster implementations."""
        return self.selection_mask(
            x=BackedTable(backend=NpTable(data=x)),
            outcomes=outcomes, printer=printer, n_proc=n_proc, covariates=covariates)

    def __str__(self) -> str:
        return "Multi-objective feature selector without name"


class FeatureSelectorMOUnion(FeatureSelectorMO):
    __feature_selector_so: ManyFeatureSelector

    def __init__(self, feature_selector_so: ManyFeatureSelector):
        self.__feature_selector_so = feature_selector_so

    def selection_mask(self, x: Table, outcomes: Sequence[Outcome], printer: Printer = NullPrinter(),
                       n_proc: int = 1, covariates: Optional[Table] = None) -> list[bool]:
        res = [False] * x.n_col()
        printer.print("Existing features: " + str(len(res)))
        printer.print("Applying union strategy for feature selection using " + str(self.__feature_selector_so))
        for o in outcomes:
            printer.print("Computing active features for outcome " + o.name())
            o_selected = self.__feature_selector_so.selection_mask(
                data=x, outcome=o, n_proc=n_proc, covariates=covariates, printer=printer)
            printer.print("Features selected for this outcome: " + str(sum(o_selected)))
            res = list_or(list_a=res, list_b=o_selected)
            printer.print("Features active in total: " + str(sum(res)))
        return res

    def selection_mask_df(
            self, x: DataFrame, outcomes: Sequence[Outcome],
            printer: Printer = NullPrinter(), n_proc: int = 1, covariates: Optional[Table] = None) -> list[bool]:
        res = [False] * len(x.columns)
        printer.print("Existing features: " + str(len(res)))
        printer.print("Applying union strategy for feature selection using " + str(self.__feature_selector_so))
        for o in outcomes:
            printer.print("Computing active features for outcome " + o.name())
            o_selected = self.__feature_selector_so.selection_mask_df(
                data=x, outcome=o, n_proc=n_proc, covariates=covariates, printer=printer)
            printer.print("Features selected for this outcome: " + str(sum(o_selected)))
            res = list_or(list_a=res, list_b=o_selected)
            printer.print("Features active in total: " + str(sum(res)))
        return res

    def __str__(self) -> str:
        return "Multi-objective feature selector with inner " + str(self.__feature_selector_so)

    def name(self) -> str:
        return "MO " + self.__feature_selector_so.name()

    def nick(self) -> str:
        return self.__feature_selector_so.nick()


class DummySelectorMO(FeatureSelectorMO):
    """Accepts every feature."""

    def selection_mask(self, x: Table, outcomes: Sequence[Outcome], printer: Printer = NullPrinter(),
                       n_proc: int = 1, covariates: Optional[Table] = None) -> list[bool]:
        return [True] * x.n_col()

    def selection_mask_df(
            self, x: DataFrame, outcomes: Sequence[Outcome],
            printer: Printer = NullPrinter(), n_proc: int = 1, covariates: Optional[Table] = None) -> list[bool]:
        return [True] * len(x.columns)

    def __str__(self) -> str:
        return "Dummy multi-objective feature selector"

    def name(self) -> str:
        return "MO dummy FS"

    def nick(self) -> str:
        return "dummy"


DUMMY_SELECTOR = DummySelectorMO()