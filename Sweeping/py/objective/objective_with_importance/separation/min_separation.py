from collections.abc import Sequence

from objective.objective_with_importance.separation.separation import Separation


class MinSeparation(Separation):

    def _separation_from_class_separations(self, class_separations: Sequence[float]) -> float:
        min_separation = None
        for separation in class_separations:
            if min_separation is None:
                min_separation = separation
            else:
                min_separation = min(min_separation, separation)
        return min_separation

    def nick(self):
        return "min_separation"

    def name(self):
        return "min separation"
