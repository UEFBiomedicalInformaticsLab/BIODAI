from collections.abc import Sequence
from math import sqrt
from objective.objective_with_importance.separation.separation import Separation
from util.math.mean_builder import KahanMeanBuilder


class RootSeparation(Separation):

    def _separation_from_class_separations(self, class_separations: Sequence[float]) -> float:
        mean_builder = KahanMeanBuilder()
        for separation in class_separations:
            mean_builder.add(sqrt(separation))
        return mean_builder.mean()

    def nick(self):
        return "root_separation"

    def name(self):
        return "root separation"