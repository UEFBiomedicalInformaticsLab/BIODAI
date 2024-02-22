from sklearn.ensemble import RandomForestClassifier
from model.model_with_coef import SklearnClassModelWrapperWithFallback, SKLearnModelFactoryWithExtractor, \
    SklearnCoefExtractor, OffCoefExtractor
from model.tree import DEFAULT_MIN_SAMPLES_LEAF

FOREST_NAME = "RF"


class ForestFactory(SKLearnModelFactoryWithExtractor):
    __min_samples_leaf: int

    def __init__(self, min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        self.__min_samples_leaf = min_samples_leaf

    def create(self):
        return RandomForestClassifier(
            class_weight="balanced", n_estimators=30, min_samples_leaf=self.__min_samples_leaf, n_jobs=1)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def min_samples_leaf(self) -> int:
        return self.__min_samples_leaf

    def supports_weights(self) -> bool:
        return True


class ForestWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self, min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        SklearnClassModelWrapperWithFallback.__init__(
            self, model_factory=ForestFactory(min_samples_leaf=min_samples_leaf))

    def min_samples_leaf(self) -> int:
        return self.model_factory().min_samples_leaf()

    def nick(self) -> str:
        return FOREST_NAME + str(self.min_samples_leaf())

    def name(self) -> str:
        return "random forest (msl" + str(self.min_samples_leaf()) + ")"

    def __str__(self) -> str:
        return self.name()
