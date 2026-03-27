import warnings
from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Optional

from numpy import ravel
from pandas import DataFrame

from model.coef_extractor import SklearnCoefExtractor
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.predictor_with_coef import SVPredictorWithCoef, SklearnPredictorWrapperWithExtractor
from model.sv_model import SVModel, SKLearnModelFactory, PredictStrategy, JustPredictStrategy
from util.table.table_utils import n_col


class SVModelWithCoef(SVModel):

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVPredictorWithCoef:
        raise NotImplementedError()


class SKLearnModelFactoryWithExtractor(SKLearnModelFactory, ABC):

    @abstractmethod
    def coef_extractor(self) -> SklearnCoefExtractor:
        raise NotImplementedError()

    def importance_extractor(self) -> SklearnImportanceExtractor:
        return OffImportanceExtractor()


class SklearnModelWrapper(SVModelWithCoef):
    __sklearn_model_factory: SKLearnModelFactoryWithExtractor
    __ignore_fit_warn: bool
    __predict_strategy: PredictStrategy

    def __init__(self,
                 model_factory: SKLearnModelFactoryWithExtractor,
                 ignore_fit_warn: bool = True,
                 predict_strategy: PredictStrategy = JustPredictStrategy()):
        self.__sklearn_model_factory = model_factory
        self.__ignore_fit_warn = ignore_fit_warn
        self.__predict_strategy = predict_strategy

    def _predict_strategy(self) -> PredictStrategy:
        return self.__predict_strategy

    def _fit_sklearn_predictor(self, x: DataFrame, y, sample_weight: Optional = None):
        model = self.__sklearn_model_factory.create()
        if model is None:
            raise ValueError("Model is None.")
        y = ravel(y)
        if self._ignore_fit_warn():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if self.__sklearn_model_factory.supports_weights():
                    sklearn_predictor = model.fit(x, y, sample_weight)
                else:
                    sklearn_predictor = model.fit(x, y)
        else:
            if self.__sklearn_model_factory.supports_weights():
                sklearn_predictor = model.fit(x, y, sample_weight)
            else:
                sklearn_predictor = model.fit(x, y)
        return sklearn_predictor

    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVPredictorWithCoef:
        sklearn_predictor = self._fit_sklearn_predictor(x=x, y=y, sample_weight=sample_weight)
        return SklearnPredictorWrapperWithExtractor(
            sklearn_predictor=sklearn_predictor, coef_extractor=self.__sklearn_model_factory.coef_extractor(),
            importance_extractor=self.__sklearn_model_factory.importance_extractor(),
            predict_strategy=self._predict_strategy())

    def nick(self) -> str:
        return self.model_factory().nick()

    def name(self) -> str:
        return self.model_factory().name()

    def __str__(self) -> str:
        return "Sklearn model wrapper with model factory " + str(self.__sklearn_model_factory)

    def model_factory(self) -> SKLearnModelFactoryWithExtractor:
        return self.__sklearn_model_factory

    def _ignore_fit_warn(self) -> bool:
        return self.__ignore_fit_warn


class SklearnModelWrapperWithFallback(SklearnModelWrapper):
    __fallback: SklearnModelWrapper
    __verbose: bool

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor,
                 fallback_model_factory: SKLearnModelFactoryWithExtractor, ignore_fit_warn: bool = True,
                 predict_strategy: PredictStrategy = JustPredictStrategy(), verbose: bool = True):
        SklearnModelWrapper.__init__(
            self=self, model_factory=model_factory, ignore_fit_warn=ignore_fit_warn, predict_strategy=predict_strategy)
        self.__fallback = SklearnModelWrapper(
            model_factory=fallback_model_factory, ignore_fit_warn=ignore_fit_warn, predict_strategy=predict_strategy)
        self.__verbose = verbose

    def fit(self, x: DataFrame, y: Sequence[float], sample_weight: Optional = None) -> SVPredictorWithCoef:
        if n_col(x) == 0:
            return self.__fallback.fit(x=x, y=y, sample_weight=sample_weight)
        else:
            try:
                return SklearnModelWrapper.fit(self=self, x=x, y=y, sample_weight=sample_weight)
            except BaseException as e:
                if self.__verbose:
                    print("There has been an exception during model fitting.")
                    print(str(e))
                    print("Retrying with fallback model.")
                return self.__fallback.fit(x=x, y=y, sample_weight=sample_weight)
                # In case of exception we fall back to dummy.

    def __str__(self) -> str:
        return "Sklearn model wrapper with model factory " + str(self.model_factory()) + " and fallback"


class SklearnModelCreator:
    __fallback_model_factory: SKLearnModelFactoryWithExtractor
    __predict_strategy: PredictStrategy

    def __init__(self,
                 fallback_model_factory: SKLearnModelFactoryWithExtractor,
                 predict_strategy: PredictStrategy):
        self.__fallback_model_factory = fallback_model_factory
        self.__predict_strategy = predict_strategy

    def create_model(self, model_factory: SKLearnModelFactoryWithExtractor) -> SVModelWithCoef:
        return SklearnModelWrapperWithFallback(
            model_factory=model_factory,
            fallback_model_factory=self.__fallback_model_factory,
            predict_strategy=self.__predict_strategy)