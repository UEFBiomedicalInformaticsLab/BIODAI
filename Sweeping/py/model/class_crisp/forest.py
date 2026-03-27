from model.classification.forest import ForestFactory, DEFAULT_FOREST_MIN_SAMPLES_LEAF
from model.class_crisp.classifier_with_coef import SklearnClassModelWrapperWithFallback


class ForestWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self, min_samples_leaf: int = DEFAULT_FOREST_MIN_SAMPLES_LEAF):
        SklearnClassModelWrapperWithFallback.__init__(
            self, model_factory=ForestFactory(min_samples_leaf=min_samples_leaf))

    def min_samples_leaf(self) -> int:
        return self.model_factory().min_samples_leaf()
