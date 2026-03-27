import math
from abc import ABC, abstractmethod

from util.math.mean_builder import MeanBuilder
from util.utils import IllegalStateError


class VarianceBuilder(MeanBuilder, ABC):

    @abstractmethod
    def tot_sum_squares(self) -> float:
        """Zero when there are no samples."""
        raise NotImplementedError()

    def has_variance(self) -> bool:
        """Has received at least 2 samples."""
        return self.num_samples() > 1

    def unbiased_variance(self) -> float:
        """Unbiased thanks to Bessel's correction.
        Note that to obtain and unbiased standard deviation it is not sufficient
        to square root the unbiased variance. Computing the unbiased standard
        deviation is dependent on the underlying distribution."""
        if self.has_variance():
            return self.tot_sum_squares() / float(self.num_samples() - 1)
        else:
            raise IllegalStateError()

    def biased_variance(self) -> float:
        """Note that to obtain and unbiased standard deviation it is not sufficient
        to square root the unbiased variance. Computing the unbiased standard
        deviation is dependent on the underlying distribution."""
        if self.has_variance():
            return self.tot_sum_squares() / float(self.num_samples())
        else:
            raise IllegalStateError()

    def biased_standard_deviation(self) -> float:
        return math.sqrt(self.biased_variance())

    def sample_standard_deviation(self) -> float:
        return math.sqrt(self.unbiased_variance())
