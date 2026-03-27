from abc import ABC, abstractmethod
from typing import Sequence, Optional

from pandas import DataFrame

from model.sv_model import SVPredictor, SklearnSVPredictorWrapper, SVModel
from util.utils import IllegalStateError


class SVRegressor(SVPredictor, ABC):

    @abstractmethod
    def predict(self, x: DataFrame) -> Sequence[float]:
        raise NotImplementedError()

    def predict_crisp(self, x: DataFrame) -> Sequence:
        raise IllegalStateError()

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        raise IllegalStateError()

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()


class RegressorSVModel(SVModel):
    """Regressor single view model."""

    @abstractmethod
    def fit(self, x: DataFrame, y: Sequence[float], sample_weight: Optional = None) -> SVRegressor:
        """ y is a Sequence of floats, each element being an expected output.
            Returns a Regressor. The model itself is not affected by the call."""
        raise NotImplementedError()


class SklearnSVRegressorWrapper(SklearnSVPredictorWrapper, SVRegressor):

    def __init__(self, sklearn_predictor):
        SklearnSVPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor)
