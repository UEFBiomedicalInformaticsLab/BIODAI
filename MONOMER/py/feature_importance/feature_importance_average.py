from collections import Sequence

from feature_importance.feature_importance import FeatureImportance
from util.distribution.average_distribution import AverageDistribution
from util.distribution.distribution import Distribution
from util.sequence_utils import str_in_lines


class FeatureImportanceAverage(FeatureImportance):
    __components: Sequence[FeatureImportance]

    def __init__(self, components: Sequence[FeatureImportance]):
        self.__components = components

    def compute(self, x, y, n_proc: int = 1) -> Distribution:
        return AverageDistribution([c.compute(x, y, n_proc=n_proc) for c in self.__components])

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
