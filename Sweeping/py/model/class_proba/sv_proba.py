from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame
from sklearn.dummy import DummyClassifier

from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.model_with_coef import SVModelWithCoef, SKLearnModelFactoryWithExtractor, SklearnModelWrapperWithFallback, \
    SklearnModelCreator
from model.coef_extractor import EmptyCoefExtractor, SklearnCoefExtractor
from model.predictor_with_coef import SVPredictorWithClassCoef, SklearnPredictorWrapperWithExtractor
from model.sv_model import SklearnSVPredictorWrapper, SVPredictor, PredictProbaStrategy, SVModel, DUMMY_STR
from util.utils import IllegalStateError


class SVProba(SVPredictor, ABC):

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        raise IllegalStateError()

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()


class SklearnSVProbaWrapper(SklearnSVPredictorWrapper, SVProba):

    def __init__(self, sklearn_predictor):
        SklearnSVPredictorWrapper.__init__(
            self, sklearn_predictor=sklearn_predictor, predict_strategy=PredictProbaStrategy())


class SVProbaWithCoef(SVProba, SVPredictorWithClassCoef, ABC):
    pass


class SklearnProbaWrapperWithCoef(SklearnSVProbaWrapper, SVProbaWithCoef, ABC):
    pass


class SklearnProbaWrapperWithExtractor(SklearnPredictorWrapperWithExtractor, SklearnProbaWrapperWithCoef):

    def __init__(self, sklearn_predictor, coef_extractor: SklearnCoefExtractor,
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor()):
        SklearnPredictorWrapperWithExtractor.__init__(
            self=self,
            sklearn_predictor=sklearn_predictor,
            coef_extractor=coef_extractor,
            importance_extractor=importance_extractor,
            predict_strategy=PredictProbaStrategy())


class ProbaSVModel(SVModel, ABC):

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVProba:
        raise NotImplementedError()


class ProbaModelWithCoef(ProbaSVModel, SVModelWithCoef, ABC):

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVProbaWithCoef:
        raise NotImplementedError()


class DummyProbaFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return DummyClassifier(strategy="prior")

    def coef_extractor(self) -> SklearnCoefExtractor:
        return EmptyCoefExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return DUMMY_STR


class SklearnProbaModelWrapper(SklearnModelWrapperWithFallback, ProbaModelWithCoef, ABC):

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor, ignore_fit_warn: bool = True):
        SklearnModelWrapperWithFallback.__init__(
            self=self, model_factory=model_factory, fallback_model_factory=DummyProbaFactory(),
            ignore_fit_warn=ignore_fit_warn, predict_strategy=PredictProbaStrategy())


class SklearnProbaClassModelCreator(SklearnModelCreator):

    def __init__(self):
        SklearnModelCreator.__init__(
            self=self, fallback_model_factory=DummyProbaFactory(), predict_strategy=PredictProbaStrategy())
