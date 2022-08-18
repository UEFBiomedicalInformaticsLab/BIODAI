import warnings
from abc import abstractmethod, ABC
from collections.abc import Sequence

from numpy import ravel
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_importance.feature_importance_by_lasso import collapse_coef
from model.model import Predictor, Model, ClassModel, Classifier, SklearnPredictorWrapper, SKLearnModelFactory, \
    DEFAULT_LOGISTIC_MAX_ITER
from util.utils import IllegalStateError


LOGISTIC_NAME = "logistic"


class PredictorWithCoef(Predictor):

    @abstractmethod
    def coef(self) -> Sequence[Sequence[float]]:
        raise NotImplementedError()

    def feature_importance(self) -> Sequence[float]:
        return collapse_coef(self.coef())


class ModelWithCoef(Model):

    @abstractmethod
    def fit(self, x, y) -> PredictorWithCoef:
        raise NotImplementedError()


class ClassifierWithCoef(Classifier, PredictorWithCoef, ABC):
    pass


class ClassModelWithCoef(ClassModel, ModelWithCoef):

    @abstractmethod
    def fit(self, x, y) -> ClassifierWithCoef:
        raise NotImplementedError()


class SklearnCoefExtractor(ABC):

    @abstractmethod
    def can_extract(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise NotImplementedError()


class OffCoefExtractor(SklearnCoefExtractor):

    def can_extract(self) -> bool:
        return False

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise IllegalStateError()


class OnCoefExtractor(SklearnCoefExtractor, ABC):

    def can_extract(self) -> bool:
        return True


class EmptyCoefExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return []


class SklearnClassifierWrapperWithCoef(SklearnPredictorWrapper, ClassifierWithCoef, ABC):
    pass


class SklearnWrapperWithExtractor(SklearnClassifierWrapperWithCoef):
    __extractor: SklearnCoefExtractor

    def __init__(self, sklearn_predictor, extractor: SklearnCoefExtractor):
        SklearnClassifierWrapperWithCoef.__init__(self, sklearn_predictor=sklearn_predictor)
        self.__extractor = extractor

    def coef(self) -> Sequence[Sequence[float]]:
        return self.__extractor.extract_coef(self._sklearn_predictor())


class SKLearnModelFactoryWithExtractor(SKLearnModelFactory, ABC):

    @abstractmethod
    def extractor(self) -> SklearnCoefExtractor:
        raise NotImplementedError()


class SklearnModelWrapper(ClassModelWithCoef, ABC):
    __sklearn_model_factory: SKLearnModelFactoryWithExtractor

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor):
        self.__sklearn_model_factory = model_factory

    def fit(self, x, y) -> ClassifierWithCoef:
        model = self.__sklearn_model_factory.create()
        sklearn_predictor = model.fit(x, ravel(y))
        return SklearnWrapperWithExtractor(
            sklearn_predictor=sklearn_predictor, extractor=self.__sklearn_model_factory.extractor())


class SklearnModelWrapperWithFallback(ClassModelWithCoef, ABC):
    __sklearn_model_factory: SKLearnModelFactoryWithExtractor
    __ignore_fit_warn: bool

    def __init__(self, model_factory: SKLearnModelFactoryWithExtractor, ignore_fit_warn: bool = True):
        self.__sklearn_model_factory = model_factory
        self.__ignore_fit_warn = ignore_fit_warn

    def fit(self, x, y) -> ClassifierWithCoef:
        if (len(x.columns)) == 0:
            sk_model = DummyClassifier(strategy="most_frequent")
            extractor = EmptyCoefExtractor()
        else:
            sk_model = self.__sklearn_model_factory.create()
            extractor = self.__sklearn_model_factory.extractor()
        y = ravel(y)
        if self.__ignore_fit_warn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                sklearn_predictor = sk_model.fit(x, y)
        else:
            sklearn_predictor = sk_model.fit(x, y)
        return SklearnWrapperWithExtractor(sklearn_predictor=sklearn_predictor, extractor=extractor)

    def model_factory(self) -> SKLearnModelFactoryWithExtractor:
        return self.__sklearn_model_factory


class NBFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        return GaussianNB()  # We do not need to standardize since gaussian nb does it internally.

    def extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()


class LogisticExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return sklearn_predictor[LOGISTIC_NAME].coef_


class LogisticFactory(SKLearnModelFactoryWithExtractor):
    __max_iter: int
    __penalty: str

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: str = 'none'):
        self.__max_iter = max_iter
        self.__penalty = penalty

    def create(self):
        solver = 'lbfgs'
        penalty = self.__penalty
        if penalty == 'l1':
            solver = 'liblinear'
        pipe = Pipeline([
            ('scale', StandardScaler()),
            (LOGISTIC_NAME, LogisticRegression(penalty=penalty, max_iter=self.__max_iter, solver=solver, n_jobs=1))])
        return pipe

    def max_iter(self) -> int:
        return self.__max_iter

    def extractor(self) -> SklearnCoefExtractor:
        return LogisticExtractor()


class NBWithFallback(SklearnModelWrapperWithFallback):

    def __init__(self):
        SklearnModelWrapperWithFallback.__init__(self, model_factory=NBFactory())

    def nick(self) -> str:
        return "NB"

    def name(self) -> str:
        return "naive Bayes"

    def __str__(self) -> str:
        return self.name()


class LogisticWithFallback(SklearnModelWrapperWithFallback):

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER, penalty: str = 'none'):
        SklearnModelWrapperWithFallback.__init__(
            self, model_factory=LogisticFactory(max_iter=max_iter, penalty=penalty))

    def max_iter(self) -> int:
        return self.model_factory().max_iter()

    def nick(self) -> str:
        return "logit" + str(self.max_iter())

    def name(self) -> str:
        return "logistic classifier (max_iter=" + str(self.max_iter()) + ")"

    def __str__(self) -> str:
        return self.name()


class LassoWithFallback(LogisticWithFallback):

    def __init__(self, max_iter: int = DEFAULT_LOGISTIC_MAX_ITER):
        LogisticWithFallback.__init__(self, max_iter=max_iter, penalty='l1')
