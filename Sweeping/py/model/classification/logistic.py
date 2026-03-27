from typing import Sequence, Optional

from sklearn.linear_model import LogisticRegression

from model.coef_extractor import OnCoefExtractor, SklearnCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor
from model.pipe_wrapper import PipeWrapper

DEFAULT_LOGISTIC_MAX_ITER = 3000
DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER = 100
LOGISTIC_NAME = "logistic"
DEFAULT_LOGISTIC_PENALTY = None


def penalty_nick(penalty: str) -> str:
    if penalty is None or penalty == 'none':
        return ""
    else:
        return penalty


def logistic_nick(penalty: str, max_iter: int) -> str:
    return penalty_nick(penalty=penalty) + "logit" + str(max_iter)


def logistic_name(penalty: str, max_iter: int) -> str:
    pnick = penalty_nick(penalty=penalty)
    if pnick is None or pnick == "":
        penalty_part = ""
    else:
        penalty_part = pnick + " regularized "
    return penalty_part + "logistic classifier (max_iter=" + str(max_iter) + ")"


class LogisticExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return sklearn_predictor[LOGISTIC_NAME].coef_


class LogisticFactory(SKLearnModelFactoryWithExtractor):
    __max_iter: int
    __penalty: Optional[str]

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY):
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

    def penalty(self) -> Optional[str]:
        return self.__penalty

    def penalty_str(self) -> str:
        if self.__penalty is None:
            return "none"
        else:
            return str(self.__penalty)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return LogisticExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return logistic_nick(penalty=self.penalty_str(), max_iter=self.max_iter())

    def name(self) -> str:
        return logistic_name(penalty=self.penalty_str(), max_iter=self.max_iter())
