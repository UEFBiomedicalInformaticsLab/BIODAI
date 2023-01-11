from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnCoefExtractor, OffCoefExtractor, \
    SklearnModelWrapperWithFallback


class SVMFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('svm', svm.SVC(kernel='rbf', class_weight='balanced'))])
        return pipe

    def extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()


class SVMWithFallback(SklearnModelWrapperWithFallback):

    def __init__(self):
        SklearnModelWrapperWithFallback.__init__(self, model_factory=SVMFactory())

    def nick(self) -> str:
        return "svm"

    def name(self) -> str:
        return "svm"

    def __str__(self) -> str:
        return self.name()
