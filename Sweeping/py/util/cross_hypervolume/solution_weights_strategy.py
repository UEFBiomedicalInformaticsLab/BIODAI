from abc import abstractmethod, ABC
from collections.abc import Sequence

from util.cross_hypervolume.hypervolume_derivatives import normalized_hypervolume_derivatives
from util.hyperbox.hyperbox import Hyperbox0B


class SolutionWeightsStrategy(ABC):

    @abstractmethod
    def assign_weights(self, hyperboxes: Sequence[Hyperbox0B]) -> list[list[float]]:
        """May contain dominated boxes.
        Returns a list for each solution, containing a list with the weight with respect to each objective."""


class UniformWeights(SolutionWeightsStrategy):

    def assign_weights(self, hyperboxes: Sequence[Hyperbox0B]) -> list[list[float]]:
        n_boxes = len(hyperboxes)
        if n_boxes == 0:
            return []
        else:
            const = 1.0/float(n_boxes)
            to_copy = [const]*hyperboxes[0].n_dimensions()
            return [to_copy]*n_boxes


class SolutionDerivatives(SolutionWeightsStrategy):

    def assign_weights(self, hyperboxes: Sequence[Hyperbox0B]) -> list[list[float]]:
        return normalized_hypervolume_derivatives(hyperboxes=hyperboxes)
