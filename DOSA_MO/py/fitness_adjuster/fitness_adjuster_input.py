from pandas import DataFrame


ORIGINAL_FITNESS_STR = "original_fitness"
BOOTSTRAP_MEAN_STR = "bootstrap_mean"
NUM_FEATURES_STR = "num_features"
STANDARD_DEVIATION_STR = "std_dev"


class FitnessAdjusterInput:
    __original_fitness: float
    __std_dev: float
    __num_features: int
    __bootstrap_mean: float

    def __init__(
            self,
            original_fitness: float,
            std_dev: float,
            num_features: int,
            bootstrap_mean: float):
        self.__original_fitness = original_fitness
        self.__std_dev = std_dev
        self.__num_features = num_features
        self.__bootstrap_mean = bootstrap_mean

    def original_fitness(self) -> float:
        return self.__original_fitness

    def std_dev(self) -> float:
        return self.__std_dev

    def num_features(self) -> int:
        return self.__num_features

    def bootstrap_mean(self) -> float:
        return self.__bootstrap_mean

    def to_df(self) -> DataFrame:
        d = {}
        d[ORIGINAL_FITNESS_STR] = [self.original_fitness()]
        d[STANDARD_DEVIATION_STR] = [self.std_dev()]
        d[NUM_FEATURES_STR] = [self.num_features()]
        d[BOOTSTRAP_MEAN_STR] = [self.bootstrap_mean()]
        return DataFrame(d)

    def __str__(self) -> str:
        return str(self.to_df())
