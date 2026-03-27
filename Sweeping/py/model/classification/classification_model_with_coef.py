from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame

from model.class_crisp.classifier_with_coef import SVClassifierWithCoef, SklearnClassifierWrapperWithExtractor
from model.coef_extractor import SklearnCoefExtractor
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnModelWrapper
from model.sv_model import PredictStrategy, JustPredictStrategy


class SKLearnClassificationModelFactoryWithExtractor(SKLearnModelFactoryWithExtractor, ABC):

    @abstractmethod
    def coef_extractor(self) -> SklearnCoefExtractor:
        raise NotImplementedError()

    def importance_extractor(self) -> SklearnImportanceExtractor:
        return OffImportanceExtractor()


class SklearnClassificationModelWrapper(SklearnModelWrapper):

    def __init__(self,
                 model_factory: SKLearnClassificationModelFactoryWithExtractor,
                 ignore_fit_warn: bool = True,
                 predict_strategy: PredictStrategy = JustPredictStrategy()):
        SklearnModelWrapper.__init__(
            self=self, model_factory=model_factory, ignore_fit_warn=ignore_fit_warn, predict_strategy=predict_strategy)

    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVClassifierWithCoef:
        sklearn_predictor = self._fit_sklearn_predictor(x=x, y=y, sample_weight=sample_weight)
        return SklearnClassifierWrapperWithExtractor(
            sklearn_predictor=sklearn_predictor, coef_extractor=self.model_factory().coef_extractor(),
            importance_extractor=self.model_factory().importance_extractor(),
            predict_strategy=self._predict_strategy())
