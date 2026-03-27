from model.classification.svm import RSVMFactory
from model.class_crisp.classifier_with_coef import SklearnClassModelWrapperWithFallback


class RSVMWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self):
        SklearnClassModelWrapperWithFallback.__init__(self, model_factory=RSVMFactory(probability=False))

    def __str__(self) -> str:
        return self.name()
