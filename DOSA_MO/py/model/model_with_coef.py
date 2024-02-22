import warnings
from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Union, Optional

from numpy import ravel
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

from feature_importance.feature_importance_by_lasso import collapse_coef
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.model import Predictor, Model, ClassModel, Classifier, SklearnClassifierWrapper, SKLearnModelFactory, \
    SklearnPredictorWrapper
from model.pipe_wrapper import PipeWrapper
from util.math.list_math import list_abs
from util.utils import IllegalStateError, name_value


NB_NICK = "NB"


class PredictorWithCoef(Predictor):

    @abstractmethod
    def coef(self) -> Union[Sequence[Sequence[float]], Sequence[float]]:
        """Classifiers have a sequence for each class, regressors just one sequence."""
        raise NotImplementedError()

    def feature_importance(self) -> Sequence[float]:
        return collapse_coef(self.coef())

    def __str__(self) -> str:
        res = "predictor with coefficients\n"
        res += self.coefs_str() + "\n"
        return res

    @abstractmethod
    def coefs_str(self) -> str:
        raise NotImplementedError()


class ModelWithCoef(Model):

    @abstractmethod
    def fit(self, x, y, sample_weight: Optional = None) -> PredictorWithCoef:
        raise NotImplementedError()


class ClassifierWithCoef(Classifier, PredictorWithCoef, ABC):

    def classifier_coefs(self) -> Sequence[Sequence[float]]:
        return self.coef()

    def coefs_str(self) -> str:
        coefs = self.classifier_coefs()
        try:
            if len(coefs) > 0:
                if len(coefs) <= 5:
                    if len(coefs[0]) < 5:
                        return name_value(name="coefficients", value=coefs)
                    else:
                        return "Many coefficients"
                return "Many coefficients"
            else:
                return "Zero coefficients"
        except BaseException:
            raise IllegalStateError("Coefficients do not work as a sequence of sequences: " + str(coefs) + "\n")


class ClassModelWithCoef(ClassModel, ModelWithCoef):

    @abstractmethod
    def fit(self, x, y, sample_weight: Optional = None) -> ClassifierWithCoef:
        raise NotImplementedError()


class SklearnCoefExtractor(ABC):

    @abstractmethod
    def can_extract_coef(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise NotImplementedError()


class OffCoefExtractor(SklearnCoefExtractor):
    """Dummy coef extractor that cannot extract."""

    def can_extract_coef(self) -> bool:
        return False

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise IllegalStateError()


class OnCoefExtractor(SklearnCoefExtractor, ABC):

    def can_extract_coef(self) -> bool:
        return True


class EmptyCoefExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return []


class SklearnPredictorWrapperWithCoef(SklearnPredictorWrapper, PredictorWithCoef, ABC):
    pass


class SklearnWrapperWithExtractor(SklearnPredictorWrapper, PredictorWithCoef, ABC):
    __coef_extractor: SklearnCoefExtractor
    __importance_extractor: SklearnImportanceExtractor

    def __init__(self, sklearn_predictor, coef_extractor: SklearnCoefExtractor,
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor()):
        SklearnPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor)
        self.__coef_extractor = coef_extractor
        self.__importance_extractor = importance_extractor

    def coef(self) -> Sequence[Sequence[float]]:
        """Raises exception if cannot extract."""
        return self.__coef_extractor.extract_coef(self._sklearn_predictor())

    def feature_importance(self) -> Sequence[float]:
        if self.__importance_extractor.can_extract_importance():
            return self.__importance_extractor.extract_importance(self._sklearn_predictor())
        else:
            if self.__coef_extractor.can_extract_coef():
                coefs = self.__coef_extractor.extract_coef(self._sklearn_predictor())
                if len(coefs) == 0:
                    return []
                else:
                    if isinstance(coefs[0], float):
                        return list_abs(coefs)
                    else:
                        return collapse_coef(self.coef())
            else:
                raise IllegalStateError()

    def importance_str(self) -> str:
        if self.__importance_extractor.can_extract_importance():
            importance = self.__importance_extractor.extract_importance(self._sklearn_predictor())
            if len(importance) > 0:
                if len(importance) <= 5:
                    return name_value(name="feature importances", value=importance)
                else:
                    return "Many feature importances."
            else:
                return "Zero feature importances."
        else:
            return "Cannot extract feature importances."

    def __str__(self) -> str:
        res = ""
        res += SklearnPredictorWrapper.__str__(self) + "\n"
        if self.__coef_extractor.can_extract_coef():
            res += self.coefs_str() + "\n"
        elif self.__importance_extractor.can_extract_importance():
            res += self.importance_str() + "\n"
        return res


class SklearnClassifierWrapperWithCoef(SklearnClassifierWrapper, ClassifierWithCoef, ABC):
    pass


class SklearnPredictorWrapperWithExtractor(SklearnWrapperWithExtractor, SklearnPredictorWrapperWithCoef, ABC):
    pass


class SklearnClassifierWrapperWithExtractor(SklearnWrapperWithExtractor, SklearnClassifierWrapperWithCoef):
    pass


class SKLearnModelFactoryWithExtractor(SKLearnModelFactory, ABC):

    @abstractmethod
    def coef_extractor(self) -> SklearnCoefExtractor:
        raise NotImplementedError()

    def importance_extractor(self) -> SklearnImportanceExtractor:
        return OffImportanceExtractor()


class SklearnModelWrapper(ModelWithCoef, ABC):
    __sklearn_model_factory: SKLearnModelFactoryWithExtractor
    __ignore_fit_warn: bool

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor, ignore_fit_warn: bool = True):
        self.__sklearn_model_factory = model_factory
        self.__ignore_fit_warn = ignore_fit_warn

    def fit(self, x, y, sample_weight: Optional = None) -> PredictorWithCoef:
        model = self.__sklearn_model_factory.create()
        if self.__sklearn_model_factory.supports_weights():
            sklearn_predictor = model.fit(x, ravel(y), sample_weight)
        else:
            sklearn_predictor = model.fit(x, ravel(y))
        return SklearnClassifierWrapperWithExtractor(
            sklearn_predictor=sklearn_predictor, coef_extractor=self.__sklearn_model_factory.coef_extractor())

    def __str__(self) -> str:
        return "Sklearn model wrapper with model factory " + str(self.__sklearn_model_factory)

    def model_factory(self) -> SKLearnModelFactoryWithExtractor:
        return self.__sklearn_model_factory

    def _ignore_fit_warn(self) -> bool:
        return self.__ignore_fit_warn


class SklearnClassModelWrapper(SklearnModelWrapper, ClassModelWithCoef, ABC):

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor):
        SklearnModelWrapper.__init__(self=self, model_factory=model_factory)

    def fit(self, x, y, sample_weight: Optional = None) -> ClassifierWithCoef:
        res = SklearnModelWrapper.fit(self, x, y, sample_weight=sample_weight)
        if isinstance(res, ClassifierWithCoef):
            return res
        else:
            raise IllegalStateError


class SklearnClassModelWrapperWithFallback(SklearnModelWrapper, ClassModelWithCoef, ABC):

    def fit(self, x, y, sample_weight: Optional = None) -> ClassifierWithCoef:
        if (len(x.columns)) == 0:
            sk_model = DummyClassifier(strategy="most_frequent")
            extractor = EmptyCoefExtractor()
            supports_weights = True
        else:
            sk_model = self.model_factory().create()
            extractor = self.model_factory().coef_extractor()
            supports_weights = self.model_factory().supports_weights()
        y = ravel(y)
        if self._ignore_fit_warn():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if supports_weights:
                    sklearn_predictor = sk_model.fit(x, y, sample_weight=sample_weight)
                else:
                    sklearn_predictor = sk_model.fit(x, y)
        else:
            if supports_weights:
                sklearn_predictor = sk_model.fit(x, y, sample_weight=sample_weight)
            else:
                sklearn_predictor = sk_model.fit(x, y)
        return SklearnClassifierWrapperWithExtractor(sklearn_predictor=sklearn_predictor, coef_extractor=extractor)


class NBFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        # We do not need to standardize since gaussian nb does it internally.
        return PipeWrapper(sklearn_model=GaussianNB(), model_name=NB_NICK, scale=False, supports_weights=True)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def supports_weights(self) -> bool:
        return True


class NBWithFallback(SklearnClassModelWrapperWithFallback):

    def __init__(self):
        SklearnClassModelWrapperWithFallback.__init__(self, model_factory=NBFactory())

    def nick(self) -> str:
        return NB_NICK

    def name(self) -> str:
        return "naive Bayes"

    def __str__(self) -> str:
        return self.name()
