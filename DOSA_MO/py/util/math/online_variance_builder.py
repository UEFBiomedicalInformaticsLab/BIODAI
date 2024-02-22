from util.math.summer import KahanSummer, Summer
from util.math.variance_builder import VarianceBuilder
from util.utils import IllegalStateError


class OnlineVarianceBuilder(VarianceBuilder):
    """According to Wikipedia:
    A numerically stable algorithm for the sample biased and unbiased variance. It also computes the mean.
    This algorithm was found by Welford, and it has been thoroughly analysed.
    Uses also Kahan summers for additional numerical stability."""
    __n: int
    __mean: Summer  # The sum in this summer is the mean of the distribution.
    __tot_sum_squares: Summer  # Total sum of squares.

    def __init__(self):
        self.__n = 0
        self.__mean = KahanSummer()
        self.__tot_sum_squares = KahanSummer()

    def add(self, x: float):
        self.__n += 1
        delta = x - self.__mean.get_sum()
        self.__mean.add(delta/float(self.__n))
        delta2 = x - self.__mean.get_sum()
        self.__tot_sum_squares.add(delta*delta2)

    def mean(self) -> float:
        if self.has_mean():
            return self.__mean.get_sum()
        else:
            raise IllegalStateError()

    def num_samples(self) -> int:
        return self.__n

    def tot_sum_squares(self) -> float:
        return self.__tot_sum_squares.get_sum()
