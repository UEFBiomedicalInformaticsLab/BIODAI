from typing import Optional

from model.classification.logistic import DEFAULT_LOGISTIC_MAX_ITER, DEFAULT_LOGISTIC_PENALTY, LogisticFactory
from model.class_crisp.classifier_with_coef import SklearnClassModelWrapperWithFallback


class LogisticWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY):
        SklearnClassModelWrapperWithFallback.__init__(
            self, model_factory=LogisticFactory(max_iter=max_iter, penalty=penalty))

    def model_factory(self) -> LogisticFactory:
        factory = SklearnClassModelWrapperWithFallback.model_factory(self=self)
        assert isinstance(factory, LogisticFactory)
        return factory

    def max_iter(self) -> int:
        return self.model_factory().max_iter()

    def penalty_str(self) -> str:
        return self.model_factory().penalty_str()


class LassoWithFallback(LogisticWithFallback):
    """This is just a logistic with a fixed l1 penalty"""

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER):
        LogisticWithFallback.__init__(self, max_iter=max_iter, penalty='l1')
