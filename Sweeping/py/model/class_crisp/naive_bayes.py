from model.class_crisp.classifier_with_coef import SklearnClassModelWrapperWithFallback
from model.classification.naive_bayes import NBFactory


class NBWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self):
        SklearnClassModelWrapperWithFallback.__init__(self, model_factory=NBFactory())
