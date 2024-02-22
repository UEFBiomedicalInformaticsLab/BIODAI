from abc import ABC, abstractmethod
from collections.abc import Sequence

from pandas import DataFrame

from input_data.outcome import Outcome
from univariate_feature_selection.feature_selector import FeatureSelector
from util.math.list_math import list_or
from util.printer.printer import Printer, NullPrinter


class FeatureSelectorMO(ABC):

    @abstractmethod
    def selection_mask(self, x: DataFrame, outcomes: Sequence[Outcome],
                       printer: Printer = NullPrinter(), n_proc: int = 1) -> list[bool]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Multi-objective feature selector without name"


class FeatureSelectorMOUnion(FeatureSelectorMO):
    __feature_selector_so: FeatureSelector

    def __init__(self, feature_selector_so: FeatureSelector):
        self.__feature_selector_so = feature_selector_so

    def selection_mask(
            self, x: DataFrame, outcomes: Sequence[Outcome],
            printer: Printer = NullPrinter(), n_proc: int = 1) -> list[bool]:
        res = [False] * len(x.columns)
        printer.print("Existing features: " + str(len(res)))
        printer.print("Applying union strategy for feature selection using " + str(self.__feature_selector_so))
        for o in outcomes:
            printer.print("Computing active features for outcome " + o.name())
            o_selected = self.__feature_selector_so.selection_mask(view=x, outcome=o, n_proc=n_proc)
            printer.print("Features selected for this outcome: " + str(sum(o_selected)))
            res = list_or(list_a=res, list_b=o_selected)
            printer.print("Features active in total: " + str(sum(res)))
        return res

    def __str__(self) -> str:
        return "Multi-objective feature selector with inner " + str(self.__feature_selector_so)


class DummySelectorMO(FeatureSelectorMO):

    def selection_mask(
            self, x: DataFrame, outcomes: Sequence[Outcome],
            printer: Printer = NullPrinter(), n_proc: int = 1) -> list[bool]:
        return [True] * len(x.columns)

    def __str__(self) -> str:
        return "Dummy multi-objective feature selector"
