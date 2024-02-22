from abc import ABC, abstractmethod
from typing import Sequence, Optional

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text

from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor, OnImportanceExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor, SklearnCoefExtractor, OnCoefExtractor, \
    OffCoefExtractor
from model.pipe_wrapper import PipeWrapper
from model.regression.regressor import SklearnRegressorModelWrapper
from model.tree import DEFAULT_MIN_SAMPLES_LEAF
from setup.allowed_names import LASSO_NAME, RIDGE_NAME

LINEAR_NAME = "linear"
SVR_NICK = "SVR"
TREE_REGRESSOR_NICK = "tree"
RFR_NICK = "RFReg"
MLP_NICK = "MLP"
KNR_NICK = "KNR"  # Regression based on k-nearest neighbors.


class JustSKLearnRegressorCreator(ABC):

    @abstractmethod
    def create(self):
        raise NotImplementedError()

    @abstractmethod
    def supports_weights(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def needs_scaling(self) -> bool:
        raise NotImplementedError()


class CompositeRegressorFactory(SKLearnModelFactoryWithExtractor):
    __regressor_creator: JustSKLearnRegressorCreator
    __regressor_nick: str
    __coef_extractor: SklearnCoefExtractor
    __importance_extractor: SklearnImportanceExtractor
    __scale: bool

    def __init__(self,
                 regressor_creator: JustSKLearnRegressorCreator,
                 regressor_nick: str,
                 coef_extractor: SklearnCoefExtractor = OffCoefExtractor(),
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor(),
                 scale: bool = True):
        self.__regressor_creator = regressor_creator
        self.__regressor_nick = regressor_nick
        self.__coef_extractor = coef_extractor
        self.__importance_extractor = importance_extractor
        self.__scale = scale

    def create(self):
        return PipeWrapper(
            sklearn_model=self.__regressor_creator.create(), model_name=self.__regressor_nick,
            supports_weights=self.__regressor_creator.supports_weights(), scale=self.__scale)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return self.__coef_extractor

    def importance_extractor(self) -> SklearnImportanceExtractor:
        return self.__importance_extractor

    def supports_weights(self) -> bool:
        return self.__regressor_creator.supports_weights()


class CompositeRegressorModelWrapper(SklearnRegressorModelWrapper):
    __nick: str
    __name: str

    def __init__(self,
                 regressor_creator: JustSKLearnRegressorCreator,
                 regressor_nick: str,
                 regressor_name: str = None,
                 coef_extractor: SklearnCoefExtractor = OffCoefExtractor(),
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor()
                 ):
        model_factory = CompositeRegressorFactory(
            regressor_creator=regressor_creator, regressor_nick=regressor_nick,
            coef_extractor=coef_extractor, importance_extractor=importance_extractor,
            scale=regressor_creator.needs_scaling())
        SklearnRegressorModelWrapper.__init__(self, model_factory=model_factory)
        self.__nick = regressor_nick
        if regressor_name is None:
            self.__name = regressor_nick
        else:
            self.__name = regressor_name

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        return self.__name


class JustZeroRegressorCreator(JustSKLearnRegressorCreator):

    def create(self):
        return DummyRegressor(strategy="constant", constant=0.0)

    def supports_weights(self) -> bool:
        """Weights are accepted and ignored."""
        return True

    def needs_scaling(self) -> bool:
        return False


class ZeroRegressor(CompositeRegressorModelWrapper):

    def __init__(self):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustZeroRegressorCreator(),
            regressor_nick="zero",
            regressor_name="zero regressor",
            coef_extractor=OffCoefExtractor()
        )


class DummyRegressorCreator(JustSKLearnRegressorCreator):
    __strategy: str

    def __init__(self, strategy: str = "mean"):
        self.__strategy = strategy

    def create(self):
        return DummyRegressor(strategy=self.__strategy, constant=0.0)

    def supports_weights(self) -> bool:
        """Weights are accepted and ignored."""
        return True

    def needs_scaling(self) -> bool:
        return False


class DummyRegressorModel(CompositeRegressorModelWrapper):

    def __init__(self, strategy: str = "mean"):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=DummyRegressorCreator(strategy=strategy),
            regressor_nick="dummy",
            regressor_name="dummy regressor",
            coef_extractor=OffCoefExtractor()
        )


class LinearExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[float]:
        return sklearn_predictor[LINEAR_NAME].coef_


class JustLinearRegressorCreator(JustSKLearnRegressorCreator):

    def create(self):
        return LinearRegression(n_jobs=1)

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return True


class Linear(CompositeRegressorModelWrapper):

    def __init__(self):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustLinearRegressorCreator(),
            regressor_nick=LINEAR_NAME,
            regressor_name="linear regression",
            coef_extractor=LinearExtractor()
        )


class LassoExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[float]:
        return sklearn_predictor[LASSO_NAME].coef_


class JustLassoRegressorCreator(JustSKLearnRegressorCreator):

    def create(self):
        return LassoCV(n_jobs=1)

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return True


class Lasso(CompositeRegressorModelWrapper):

    def __init__(self):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustLassoRegressorCreator(),
            regressor_nick=LASSO_NAME,
            coef_extractor=LassoExtractor()
        )


class JustSVRegressorCreator(JustSKLearnRegressorCreator):
    __c: float
    __epsilon: float

    def __init__(self, c: float = 1.0, epsilon: float = 0.1):
        """c is the regularization parameter.
        The strength of the regularization is inversely proportional to c. Must be strictly positive.
        The penalty is a squared l2 penalty."""
        self.__c = c
        self.__epsilon = epsilon

    def create(self):
        return SVR(C=self.__c, epsilon=self.__epsilon)

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return True


class SVRegressor(CompositeRegressorModelWrapper):

    def __init__(self, c: float = 1.0, epsilon: float = 0.1):
        """c is the regularization parameter.
        The strength of the regularization is inversely proportional to c. Must be strictly positive.
        The penalty is a squared l2 penalty."""
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustSVRegressorCreator(c=c, epsilon=epsilon),
            regressor_nick=SVR_NICK,
            regressor_name="support vector regression"
        )


class SKLearnTreeModelDecorator:
    """Adds pretty printing of tree rules."""
    __model: DecisionTreeRegressor

    def __init__(self, model: DecisionTreeRegressor):
        self.__model = model

    def fit(self, x, y: Sequence[float], sample_weight: Optional[Sequence[float]] = None):
        self.__model.fit(X=x, y=y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.__model.predict(x)

    def feature_importances(self):
        return self.__model.feature_importances_

    def __str__(self) -> str:
        try:
            return export_text(self.__model, feature_names=[str(f) for f in self.__model.feature_names_in_])
        except AttributeError:
            return export_text(self.__model)


class JustTreeRegressorCreator(JustSKLearnRegressorCreator):
    __criterion: str
    __ccp_alpha: float
    __min_samples_leaf: int

    def __init__(self, criterion: str, ccp_alpha: float = 0.0, min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        self.__criterion = criterion
        self.__ccp_alpha = ccp_alpha
        self.__min_samples_leaf = min_samples_leaf

    def create(self):
        return SKLearnTreeModelDecorator(model=DecisionTreeRegressor(
            criterion=self.__criterion,
            ccp_alpha=self.__ccp_alpha,
            min_samples_leaf=self.__min_samples_leaf))

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return False


class TreeRegressorImportanceExtractor(OnImportanceExtractor):

    def extract_importance(self, sklearn_predictor) -> Sequence[float]:
        return sklearn_predictor[TREE_REGRESSOR_NICK].feature_importances()


class TreeRegressor(CompositeRegressorModelWrapper):

    def __init__(self, criterion: str = "squared_error",
                 ccp_alpha: float = 0.0,
                 min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustTreeRegressorCreator(
                criterion=criterion,
                ccp_alpha=ccp_alpha,
                min_samples_leaf=min_samples_leaf),
            regressor_nick=TREE_REGRESSOR_NICK,
            regressor_name="tree regression",
            importance_extractor=TreeRegressorImportanceExtractor()
        )


class JustRFRegressorCreator(JustSKLearnRegressorCreator):
    __criterion: str

    def __init__(self, criterion: str = "squared_error"):
        self.__criterion = criterion

    def create(self):
        return RandomForestRegressor(criterion=self.__criterion)

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return False


class RFRegressorImportanceExtractor(OnImportanceExtractor):

    def extract_importance(self, sklearn_predictor) -> Sequence[float]:
        return sklearn_predictor[RFR_NICK].feature_importances_


class RFRegressor(CompositeRegressorModelWrapper):

    def __init__(self, criterion: str = "squared_error"):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustRFRegressorCreator(criterion=criterion),
            regressor_nick=RFR_NICK,
            regressor_name="random forest regressor",
            importance_extractor=RFRegressorImportanceExtractor()
        )


class RidgeExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[float]:
        return sklearn_predictor[RIDGE_NAME].coef_


class JustRidgeRegressorCreator(JustSKLearnRegressorCreator):

    def create(self):
        return RidgeCV()

    def supports_weights(self) -> bool:
        return True

    def needs_scaling(self) -> bool:
        return True


class Ridge(CompositeRegressorModelWrapper):

    def __init__(self):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustRidgeRegressorCreator(),
            regressor_nick=RIDGE_NAME,
            coef_extractor=RidgeExtractor()
        )


class JustMLPRegressorCreator(JustSKLearnRegressorCreator):
    __alpha: float

    def __init__(self, alpha: float):
        self.__alpha = alpha

    def create(self) -> RegressorMixin:
        return MLPRegressor(solver="lbfgs", alpha=self.__alpha, hidden_layer_sizes=(100,), activation="relu")

    def supports_weights(self) -> bool:
        return False

    def needs_scaling(self) -> bool:
        return True


class MLPRegressorModel(CompositeRegressorModelWrapper):

    def __init__(self, alpha: float = 0.0001):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustMLPRegressorCreator(alpha=alpha),
            regressor_nick=MLP_NICK,
            coef_extractor=OffCoefExtractor()
        )


class JustKNRCreator(JustSKLearnRegressorCreator):
    __n_neighbors: int

    def __init__(self, n_neighbors: int = 5):
        self.__n_neighbors = n_neighbors

    def create(self) -> RegressorMixin:
        return KNeighborsRegressor(n_neighbors=self.__n_neighbors, n_jobs=1)

    def supports_weights(self) -> bool:
        return False

    def needs_scaling(self) -> bool:
        return True


class KNRModel(CompositeRegressorModelWrapper):

    def __init__(self, n_neighbors: int = 5):
        CompositeRegressorModelWrapper.__init__(
            self,
            regressor_creator=JustKNRCreator(n_neighbors=n_neighbors),
            regressor_nick=KNR_NICK,
            coef_extractor=OffCoefExtractor()
        )
