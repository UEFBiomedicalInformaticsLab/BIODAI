from collections.abc import Sequence

from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.summer import KahanSummer


class AverageDistribution(Distribution):
    __dists: Sequence[Distribution]

    def __init__(self, distributions: Sequence[Distribution]):
        self.__dists = distributions

    def __getitem__(self, i: int) -> float:
        return KahanSummer.mean([d[i] for d in self.__dists])

    def __len__(self) -> int:
        return len(self.__dists[0])

    def compile(self) -> Distribution:
        return ConcreteDistribution(probs=self)
