from sklearn import svm

from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnCoefExtractor, OffCoefExtractor, \
    SklearnClassModelWrapperWithFallback
from model.pipe_wrapper import PipeWrapper


SVM_NICK = "svm"


class SVMFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return PipeWrapper(sklearn_model=svm.SVC(kernel='rbf', class_weight='balanced'),
                           model_name=SVM_NICK,
                           scale=True,
                           supports_weights=False)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def supports_weights(self) -> bool:
        """Not supported at the moment. Support might be provided in the future."""
        return False


class SVMWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self):
        SklearnClassModelWrapperWithFallback.__init__(self, model_factory=SVMFactory())

    def nick(self) -> str:
        return SVM_NICK

    def name(self) -> str:
        return SVM_NICK

    def __str__(self) -> str:
        return self.name()
