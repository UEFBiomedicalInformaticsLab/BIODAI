from sklearn.ensemble import RandomForestClassifier
from model.model_with_coef import SklearnModelWrapperWithFallback, SKLearnModelFactoryWithExtractor, \
    SklearnCoefExtractor, OffCoefExtractor


class ForestFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return RandomForestClassifier(class_weight="balanced", n_estimators=30, min_samples_leaf=2, n_jobs=1)

    def extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()


class ForestWithFallback(SklearnModelWrapperWithFallback):

    def __init__(self):
        SklearnModelWrapperWithFallback.__init__(self, model_factory=ForestFactory())

    def nick(self) -> str:
        return "RF"

    def name(self) -> str:
        return "random forest"

    def __str__(self) -> str:
        return self.name()
