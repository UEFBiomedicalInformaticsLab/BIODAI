from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame

from model.coef_extractor import SklearnCoefExtractor
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.model_with_coef import SVModelWithCoef, SKLearnModelFactoryWithExtractor, \
    SklearnModelWrapperWithFallback, SklearnModelCreator
from model.predictor_with_coef import SVPredictorWithClassCoef, SklearnPredictorWrapperWithExtractor, \
    SVPredictorWithCoef
from model.class_crisp.sv_classifier import SVClassifier, ClassSVModel, SklearnSVClassifierWrapper, \
    DummyClassifierFactory
from model.sv_model import JustPredictStrategy, PredictStrategy


class SVClassifierWithCoef(SVClassifier, SVPredictorWithClassCoef, ABC):
    pass


class ClassModelWithCoef(ClassSVModel, SVModelWithCoef):

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVClassifierWithCoef:
        raise NotImplementedError()


class SklearnClassifierWrapperWithCoef(SklearnSVClassifierWrapper, SVClassifierWithCoef, ABC):
    pass


class SklearnClassifierWrapperWithExtractor(SklearnPredictorWrapperWithExtractor, SklearnClassifierWrapperWithCoef):

    def __init__(self, sklearn_predictor, coef_extractor: SklearnCoefExtractor,
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor(),
                 predict_strategy: PredictStrategy = JustPredictStrategy()):
        SklearnPredictorWrapperWithExtractor.__init__(
            self=self,
            sklearn_predictor=sklearn_predictor,
            coef_extractor=coef_extractor,
            importance_extractor=importance_extractor,
            predict_strategy=predict_strategy)


class SklearnClassModelWrapperWithFallback(SklearnModelWrapperWithFallback, ClassModelWithCoef, ABC):

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor,
                 ignore_fit_warn: bool = True):
        SklearnModelWrapperWithFallback.__init__(
            self=self,
            model_factory=model_factory,
            fallback_model_factory=DummyClassifierFactory(),
            ignore_fit_warn=ignore_fit_warn)

    def fit(self, x: DataFrame, y: DataFrame, sample_weight: Optional[DataFrame] = None) -> SVPredictorWithCoef:
        return SklearnModelWrapperWithFallback.fit(self, x=x, y=y, sample_weight=sample_weight)


class SklearnCrispClassModelCreator(SklearnModelCreator):

    def __init__(self):
        SklearnModelCreator.__init__(
            self=self, fallback_model_factory=DummyClassifierFactory(), predict_strategy=JustPredictStrategy())