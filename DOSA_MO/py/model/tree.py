from sklearn.tree import DecisionTreeClassifier

from model.model_with_coef import SklearnClassModelWrapperWithFallback, SKLearnModelFactoryWithExtractor, \
    SklearnCoefExtractor, OffCoefExtractor


DEFAULT_MIN_SAMPLES_LEAF = 4
TREE_NAME = "tree"


class TreeFactory(SKLearnModelFactoryWithExtractor):
    __min_samples_leaf: int

    def __init__(self, min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        self.__min_samples_leaf = min_samples_leaf

    def create(self):
        return DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=self.__min_samples_leaf)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def min_samples_leaf(self) -> int:
        return self.__min_samples_leaf

    def supports_weights(self) -> bool:
        return True


class TreeWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self, min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        SklearnClassModelWrapperWithFallback.__init__(
            self, model_factory=TreeFactory(min_samples_leaf=min_samples_leaf))

    def min_samples_leaf(self) -> int:
        return self.model_factory().min_samples_leaf()

    def nick(self) -> str:
        return TREE_NAME + str(self.min_samples_leaf())

    def name(self) -> str:
        return "decision tree (msl" + str(self.min_samples_leaf()) + ")"

    def __str__(self) -> str:
        return self.name()
