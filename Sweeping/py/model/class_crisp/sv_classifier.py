from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame
from sklearn.dummy import DummyClassifier

from model.coef_extractor import SklearnCoefExtractor, EmptyCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor

from model.sv_model import SVPredictor, SVModel, SklearnSVPredictorWrapper, DUMMY_STR
from util.utils import IllegalStateError


class SVClassifier(SVPredictor, ABC):

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        raise IllegalStateError("Called object is of class " + str(self.__class__))


class ClassSVModel(SVModel):
    """Abstract class for models able to learn creating a predictor."""

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVClassifier:
        """ y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call.
            Weights are optional. If they are provided by the model does not support them, they are ignored."""
        raise NotImplementedError()


class TunableSVClassModel(ABC):
    """A model that has tunable hyperparameters."""

    @abstractmethod
    def tune(self, hyperparameters) -> ClassSVModel:
        raise NotImplementedError()


class SklearnSVClassifierWrapper(SklearnSVPredictorWrapper, SVClassifier):

    def __init__(self, sklearn_predictor):
        SklearnSVPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor)


class DummyClassifierFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return DummyClassifier(strategy="most_frequent")

    def coef_extractor(self) -> SklearnCoefExtractor:
        return EmptyCoefExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return DUMMY_STR
