from typing import Sequence

from sklearn.linear_model import LogisticRegression

from model.model import DEFAULT_LOGISTIC_MAX_ITER
from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnCoefExtractor, \
    SklearnClassModelWrapperWithFallback, OnCoefExtractor
from model.pipe_wrapper import PipeWrapper

LOGISTIC_NAME = "logistic"
DEFAULT_LOGISTIC_PENALTY = "none"


class LogisticExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return sklearn_predictor[LOGISTIC_NAME].coef_


class LogisticFactory(SKLearnModelFactoryWithExtractor):

    __max_iter: int
    __penalty: str

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: str = DEFAULT_LOGISTIC_PENALTY):
        self.__max_iter = max_iter
        self.__penalty = penalty

    def create(self):
        solver = 'lbfgs'
        penalty = self.__penalty
        if penalty == 'l1':
            solver = 'liblinear'
        return PipeWrapper(
            sklearn_model=LogisticRegression(penalty=penalty, max_iter=self.__max_iter, solver=solver, n_jobs=1),
            model_name=LOGISTIC_NAME,
            supports_weights=True,
            scale=True)

    def max_iter(self) -> int:
        return self.__max_iter

    def penalty(self) -> str:
        return self.__penalty

    def coef_extractor(self) -> SklearnCoefExtractor:
        return LogisticExtractor()

    def supports_weights(self) -> bool:
        return True


class LogisticWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: str = DEFAULT_LOGISTIC_PENALTY):
        SklearnClassModelWrapperWithFallback.__init__(
            self, model_factory=LogisticFactory(max_iter=max_iter, penalty=penalty))

    def max_iter(self) -> int:
        return self.model_factory().max_iter()

    def __penalty_nick(self) -> str:
        penalty = self.model_factory().penalty()
        if penalty is None or penalty == 'none':
            return ""
        else:
            return penalty

    def nick(self) -> str:
        return self.__penalty_nick() + "logit" + str(self.max_iter())

    def name(self) -> str:
        pnick = self.__penalty_nick()
        if pnick is None or pnick == "":
            penalty_part = ""
        else:
            penalty_part = pnick + " regularized "
        return penalty_part + "logistic classifier (max_iter=" + str(self.max_iter()) + ")"

    def __str__(self) -> str:
        return self.name()


class LassoWithFallback(LogisticWithFallback):
    """This is just a logistic with a fixed l1 penalty"""

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER):
        LogisticWithFallback.__init__(self, max_iter=max_iter, penalty='l1')
