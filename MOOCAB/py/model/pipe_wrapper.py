from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCALER_NICK = "scale"


def sklearn_imputed_model(sklearn_model, model_name: str) -> Pipeline:
    imputer = SimpleImputer()
    pipe = Pipeline([("impute", imputer), (model_name, sklearn_model)])
    return pipe


class PipeWrapper:
    __supports_weights: bool
    __model_name: str
    __scale: bool

    def __init__(self,
                 sklearn_model,
                 scale: bool,
                 model_name: str,
                 supports_weights: bool):
        self.__supports_weights = supports_weights
        self.__model_name = model_name
        self.__scale = scale
        self.__sklearn_model = sklearn_model

    def fit(self, x, y, sample_weight: Optional = None):
        """The wrapped pipeline may be modified by the call."""
        if self.__scale:
            pipe = Pipeline([
                ('impute', SimpleImputer()),
                (SCALER_NICK, StandardScaler()),
                (self.__model_name, self.__sklearn_model)])
        else:
            pipe = sklearn_imputed_model(sklearn_model=self.__sklearn_model, model_name=self.__model_name)
        if self.__supports_weights:
            if self.__scale:
                return pipe.fit(X=x, y=y, **{
                    SCALER_NICK+"__sample_weight": sample_weight,
                    self.__model_name+"__sample_weight": sample_weight})
            else:
                return pipe.fit(X=x, y=y, **{self.__model_name + "__sample_weight": sample_weight})
        else:
            return pipe.fit(X=x, y=y)
