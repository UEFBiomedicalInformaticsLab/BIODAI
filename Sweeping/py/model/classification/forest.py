from sklearn.ensemble import RandomForestClassifier

from model.coef_extractor import SklearnCoefExtractor, OffCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor

FOREST_NAME = "RF"
DEFAULT_N_ESTIMATORS = 7  # Previously 30, but was depleting memory on some architectures.
DEFAULT_FOREST_MIN_SAMPLES_LEAF = 30  # Previously 4, increased to save memory.


def forest_nick(min_samples_leaf: int) -> str:
    return FOREST_NAME + str(min_samples_leaf)


def forest_name(min_samples_leaf: int) -> str:
    return "random forest (msl" + str(min_samples_leaf) + ")"


class ForestFactory(SKLearnModelFactoryWithExtractor):
    __min_samples_leaf: int
    __n_estimators: int

    def __init__(self, min_samples_leaf: int = DEFAULT_FOREST_MIN_SAMPLES_LEAF,
                 n_estimators: int = DEFAULT_N_ESTIMATORS):
        self.__min_samples_leaf = min_samples_leaf
        self.__n_estimators = n_estimators

    def create(self):
        return RandomForestClassifier(
            class_weight="balanced",
            n_estimators=self.__n_estimators, min_samples_leaf=self.__min_samples_leaf, n_jobs=1)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def min_samples_leaf(self) -> int:
        return self.__min_samples_leaf

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return forest_nick(min_samples_leaf=self.__min_samples_leaf)

    def name(self) -> str:
        return forest_name(min_samples_leaf=self.__min_samples_leaf)
