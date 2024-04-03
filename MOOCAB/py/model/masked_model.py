from typing import Sequence, Union, Optional

from pandas import DataFrame

from model.mask import Mask
from model.model_with_coef import ClassModelWithCoef, ClassifierWithCoef
from util.utils import IllegalStateError


class MaskedClassifierWithCoef(ClassifierWithCoef):
    __inner: ClassifierWithCoef
    __mask: Mask

    def __init__(self, inner: ClassifierWithCoef, mask: Mask):
        self.__inner = inner
        self.__mask = mask

    def predict(self, x: DataFrame):
        return self.__inner.predict(x=self.__mask.apply_df(df=x))

    def feature_importance(self) -> Sequence[float]:
        return self.__mask.apply_backward_seq(seq=self.__inner.feature_importance())

    def coef(self) -> Union[Sequence[Sequence[float]], Sequence[float]]:
        return self.__inner.coef()

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()



class MaskedClassModel(ClassModelWithCoef):
    __inner: ClassModelWithCoef
    __mask: Mask

    def __init__(self, inner: ClassModelWithCoef, mask: Mask):
        self.__inner = inner
        self.__mask = mask

    def fit(self, x, y, sample_weight: Optional = None) -> MaskedClassifierWithCoef:
        inner_predictor = self.__inner.fit(x=self.__mask.apply_df(df=x), y=y, sample_weight=sample_weight)
        return MaskedClassifierWithCoef(inner=inner_predictor, mask=self.__mask)

    def nick(self) -> str:
        return "masked_" + self.__inner.nick()
