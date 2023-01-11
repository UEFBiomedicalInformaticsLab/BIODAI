from typing import Sequence

from pandas import DataFrame

from model.mask import Mask
from model.model_with_coef import ClassModelWithCoef, ClassifierWithCoef


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


class MaskedClassModel(ClassModelWithCoef):
    __inner: ClassModelWithCoef
    __mask: Mask

    def __init__(self, inner: ClassModelWithCoef, mask: Mask):
        self.__inner = inner
        self.__mask = mask

    def fit(self, x, y) -> MaskedClassifierWithCoef:
        inner_predictor = self.__inner.fit(x=self.__mask.apply_df(df=x), y=y)
        return MaskedClassifierWithCoef(inner=inner_predictor, mask=self.__mask)

    def nick(self) -> str:
        return "masked_" + self.__inner.nick()
