from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pandas import DataFrame
from sklearn.dummy import DummyRegressor

from model.coef_extractor import SklearnCoefExtractor, EmptyCoefExtractor
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnModelWrapper, SVModelWithCoef, \
    SklearnModelWrapperWithFallback
from model.predictor_with_coef import SVPredictorWithCoef, SklearnPredictorWrapperWithExtractor
from model.regression.svregressor import SVRegressor, SklearnSVRegressorWrapper, \
    RegressorSVModel
from model.sv_model import PredictStrategy, JustPredictStrategy, DUMMY_STR
from util.str_utils import name_value
from util.utils import IllegalStateError


class SKLearnRegressionModelFactoryWithExtractor(SKLearnModelFactoryWithExtractor, ABC):

    @abstractmethod
    def coef_extractor(self) -> SklearnCoefExtractor:
        raise NotImplementedError()

    def importance_extractor(self) -> SklearnImportanceExtractor:
        return OffImportanceExtractor()


class SVRegressorWithCoef(SVRegressor, SVPredictorWithCoef, ABC):

    def regressor_coefs(self) -> Sequence[Sequence[float]]:
        return self.coef()

    def coefs_str(self) -> str:
        coefs = self.regressor_coefs()
        try:
            if len(coefs) > 0:
                if len(coefs) <= 5:
                    return name_value(name="coefficients", value=coefs)
                else:
                    return "Many coefficients"
            else:
                return "Zero coefficients"
        except BaseException:
            raise IllegalStateError("Coefficients do not work as a sequence: " + str(coefs) + "\n")


class SklearnRegressionModelWrapper(SklearnModelWrapper):

    def __init__(self,
                 model_factory: SKLearnRegressionModelFactoryWithExtractor,
                 ignore_fit_warn: bool = True,
                 predict_strategy: PredictStrategy = JustPredictStrategy()):
        SklearnModelWrapper.__init__(
            self=self, model_factory=model_factory, ignore_fit_warn=ignore_fit_warn, predict_strategy=predict_strategy)

    def fit(self, x: DataFrame, y, sample_weight: Optional = None) -> SVRegressorWithCoef:
        sklearn_predictor = self._fit_sklearn_predictor(x=x, y=y, sample_weight=sample_weight)
        return SklearnRegressorWrapperWithExtractor(
            sklearn_predictor=sklearn_predictor, coef_extractor=self.model_factory().coef_extractor(),
            importance_extractor=self.model_factory().importance_extractor())


class SklearnRegressorWrapperWithCoef(SklearnSVRegressorWrapper, SVRegressorWithCoef, ABC):
    pass


class RegressorModelWithCoef(RegressorSVModel, SVModelWithCoef):

    @abstractmethod
    def fit(self, x: DataFrame, y: Sequence[float], sample_weight: Optional = None) -> SVRegressorWithCoef:
        raise NotImplementedError()


class SklearnRegressorWrapperWithExtractor(SklearnPredictorWrapperWithExtractor, SklearnRegressorWrapperWithCoef):

    def __init__(self, sklearn_predictor, coef_extractor: SklearnCoefExtractor,
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor()):
        SklearnPredictorWrapperWithExtractor.__init__(
            self=self,
            sklearn_predictor=sklearn_predictor,
            coef_extractor=coef_extractor,
            importance_extractor=importance_extractor,
            predict_strategy=JustPredictStrategy())


class SklearnRegressorModelWrapper(SklearnModelWrapperWithFallback, RegressorModelWithCoef):

    def __init__(self, model_factory: SKLearnRegressionModelFactoryWithExtractor, ignore_fit_warn: bool = True):
        SklearnModelWrapperWithFallback.__init__(
            self=self,
            model_factory=model_factory,
            fallback_model_factory=DummyRegressorFactory(),
            ignore_fit_warn = ignore_fit_warn)

    def fit(self, x: DataFrame, y: Sequence[float], sample_weight: Optional = None) -> SVPredictorWithCoef:
        return SklearnModelWrapperWithFallback.fit(self=self, x=x, y=y, sample_weight=sample_weight)


class DummyRegressorFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return DummyRegressor()

    def coef_extractor(self) -> SklearnCoefExtractor:
        return EmptyCoefExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return DUMMY_STR
