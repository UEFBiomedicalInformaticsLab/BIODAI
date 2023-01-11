import random
from abc import ABC, abstractmethod

from scipy.stats import binom

from util.named import NickNamed


DEFAULT_INITIAL_FEATURES_MIN = 0
DEFAULT_INITIAL_FEATURES_MAX = 30


def binomial_extraction(n, p):
    return binom.rvs(n=n, p=p, size=1)[0]


class NumFeatures(NickNamed, ABC):

    @abstractmethod
    def extract(self, individual_size: int) -> int:
        raise NotImplementedError()


class BinomialNumFeatures(NumFeatures):
    __average_num_features: float

    def __init__(self, average_num_features: float = 15):
        self.__average_num_features = average_num_features

    def extract(self, individual_size: int) -> int:
        return binomial_extraction(n=individual_size, p=min(1.0, self.__average_num_features / individual_size))

    def nick(self) -> str:
        return "bin" + str(self.__average_num_features)

    def name(self) -> str:
        return "binomial(" + str(self.__average_num_features) + ")"

    def __str__(self) -> str:
        return "binomial with mean " + str(self.__average_num_features)


class UniformNumFeatures(NumFeatures):
    __min_num_features: int
    __max_num_features: int

    def __init__(self,
                 min_num_features: int = DEFAULT_INITIAL_FEATURES_MIN,
                 max_num_features: int = DEFAULT_INITIAL_FEATURES_MAX):
        self.__min_num_features = min_num_features
        self.__max_num_features = max_num_features

    def extract(self, individual_size: int) -> int:
        return random.randint(
            a=self.__min_num_features, b=min(self.__max_num_features, individual_size))  # b is included.

    def nick(self) -> str:
        return "uni" + str(self.__min_num_features) + "-" + str(self.__max_num_features)

    def name(self) -> str:
        return "uniform(" + str(self.__min_num_features) + ", " + str(self.__max_num_features) + ")"

    def __str__(self) -> str:
        return "uniform distribution in [" + str(self.__min_num_features) + ", " + str(self.__max_num_features) + "]"


class BinomialFromUniformNumFeatures(NumFeatures):
    """Created for retrocompatibility."""
    __uniform: UniformNumFeatures

    def __init__(self,
                 min_num_features: int = DEFAULT_INITIAL_FEATURES_MIN,
                 max_num_features: int = DEFAULT_INITIAL_FEATURES_MAX):
        """max_num_features is the max of the uniform distribution. After applying the binomial a higher value
        can be obtained."""
        self.__uniform = UniformNumFeatures(min_num_features=min_num_features, max_num_features=max_num_features)

    def extract(self, individual_size: int) -> int:
        average_features = self.__uniform.extract(individual_size)
        if individual_size == 0 or average_features == 0:
            return 0
        else:
            return binomial_extraction(n=individual_size, p=average_features / individual_size)

    def nick(self) -> str:
        return "bin(" + self.__uniform.nick() + ")"

    def name(self) -> str:
        return "binomial(" + self.__uniform.name() + ")"

    def __str__(self) -> str:
        return "binomial distribution with mean from " + str(self.__uniform)
