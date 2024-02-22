import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Optional

from numpy import ravel
from pandas import DataFrame
from sklearn.dummy import DummyRegressor

from model.importance_extractor import OffImportanceExtractor
from model.model import Predictor, SklearnPredictorWrapper, Model
from model.model_with_coef import SklearnModelWrapper, SKLearnModelFactoryWithExtractor, PredictorWithCoef, \
    ModelWithCoef, SklearnWrapperWithExtractor, OffCoefExtractor
from util.dataframes import n_col
from util.utils import IllegalStateError, name_value


class Regressor(Predictor, ABC):

    def is_class_predictor(self) -> bool:
        return False

    @abstractmethod
    def predict(self, x) -> Sequence[float]:
        raise NotImplementedError()

    def score_concordance_index(self, x_test, y_test) -> float:
        raise IllegalStateError()

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()


class RegressorModel(Model):

    @abstractmethod
    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> Regressor:
        """ x is a Sequence of anything, each element being a sample.
            y is a Sequence of floats, each element being an expected output.
            Returns a Regressor. The model itself is not affected by the call."""
        raise NotImplementedError()

    def is_class_model(self) -> bool:
        return False


class SklearnRegressorWrapper(SklearnPredictorWrapper, Regressor):

    def __init__(self, sklearn_predictor):
        SklearnPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor)


class RegressorWithCoef(Regressor, PredictorWithCoef, ABC):

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


class SklearnRegressorWrapperWithCoef(SklearnRegressorWrapper, RegressorWithCoef, ABC):
    pass


class SklearnRegressorWrapperWithExtractor(SklearnWrapperWithExtractor, SklearnRegressorWrapperWithCoef):
    pass


class RegressorModelWithCoef(RegressorModel, ModelWithCoef):

    @abstractmethod
    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> RegressorWithCoef:
        raise NotImplementedError()


class DummyRegressorModel(RegressorModelWithCoef):

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> RegressorWithCoef:
        return SklearnRegressorWrapperWithExtractor(
            sklearn_predictor=DummyRegressor().fit(x, y, sample_weight=sample_weight),
            coef_extractor=OffCoefExtractor(),
            importance_extractor=OffImportanceExtractor())

    def nick(self) -> str:
        return "dummy"


class SklearnRegressorModelWrapper(SklearnModelWrapper, RegressorModelWithCoef, ABC):
    __fallback: bool

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor, fallback: bool = True):
        SklearnModelWrapper.__init__(self=self, model_factory=model_factory)
        self.__fallback = fallback

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> RegressorWithCoef:
        if n_col(x) > 0:
            sk_model = self.model_factory().create()
            coef_extractor = self.model_factory().coef_extractor()
            importance_extractor = self.model_factory().importance_extractor()
            y = ravel(y)
            try:
                if self._ignore_fit_warn():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if self.model_factory().supports_weights():
                            sklearn_predictor = sk_model.fit(x, y, sample_weight=sample_weight)
                        else:
                            sklearn_predictor = sk_model.fit(x, y)
                else:
                    if self.model_factory().supports_weights():
                        sklearn_predictor = sk_model.fit(x, y, sample_weight=sample_weight)
                    else:
                        sklearn_predictor = sk_model.fit(x, y)
            except BaseException as e:
                if self.__fallback:
                    return DummyRegressorModel().fit(x, y, sample_weight=sample_weight)
                    # In case of exception we fall back to dummy.
                else:
                    raise e
            res = SklearnRegressorWrapperWithExtractor(
                sklearn_predictor=sklearn_predictor,
                coef_extractor=coef_extractor, importance_extractor=importance_extractor)
            if isinstance(res, RegressorWithCoef):
                return res
            else:
                raise IllegalStateError(
                    "Learned predictor is not a regressor with coefficients.\n" +
                    "Learned predictor:\n" +
                    str(res) + "\n" +
                    "self:\n" +
                    str(self))
        else:
            return DummyRegressorModel().fit(x, y, sample_weight=sample_weight)
