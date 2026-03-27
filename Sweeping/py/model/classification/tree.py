from sklearn.tree import DecisionTreeClassifier

from model.coef_extractor import SklearnCoefExtractor, OffCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor

DEFAULT_TREE_MIN_SAMPLES_LEAF = 4
TREE_NAME = "tree"


def tree_nick(min_samples_leaf: int) -> str:
    return TREE_NAME + str(min_samples_leaf)


def tree_name(min_samples_leaf: int) -> str:
    return "decision tree (msl" + str(min_samples_leaf) + ")"


class TreeFactory(SKLearnModelFactoryWithExtractor):
    __min_samples_leaf: int

    def __init__(self, min_samples_leaf: int = DEFAULT_TREE_MIN_SAMPLES_LEAF):
        self.__min_samples_leaf = min_samples_leaf

    def create(self):
        return DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=self.__min_samples_leaf)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def min_samples_leaf(self) -> int:
        return self.__min_samples_leaf

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return tree_nick(min_samples_leaf=self.min_samples_leaf())

    def name(self) -> str:
        return tree_name(min_samples_leaf=self.min_samples_leaf())
