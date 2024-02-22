from abc import abstractmethod

from util.distribution.distribution import Distribution
from util.named import NickNamed


class FeatureImportance(NickNamed):

    @abstractmethod
    def compute(self, x, y, n_proc: int = 1) -> Distribution:
        """Returned list assigns an importance to each feature. x should be a single view."""
        raise NotImplementedError()
