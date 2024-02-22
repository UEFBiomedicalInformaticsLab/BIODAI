from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from util.named import NickNamed, Named
from util.utils import IllegalStateError, name_value

DEFAULT_LOGISTIC_MAX_ITER = 3000
DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER = 100


# TODO Split completely single and multi view models and predictors.


class Predictor:

    @abstractmethod
    def is_class_predictor(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x) -> Sequence:
        """x is a list of anything, each element being a sample.
        Returns a list of anything, each element being an expected output.
        x might be a DataFrame."""
        raise NotImplementedError()

    @abstractmethod
    def score_concordance_index(self, x_test, y_test) -> float:
        raise NotImplementedError()

    @abstractmethod
    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        """Return probabilities that event has not happened up to the passed times.
        It returns times on the rows and individuals on the columns."""
        raise NotImplementedError()


class Model(NickNamed):

    @abstractmethod
    def fit(self, x, y, sample_weight: Optional = None) -> Predictor:
        """ x is a list of anything, each element being a sample.
            y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call.
            Weights are optional. If they are provided by the model does not support them, they are ignored."""
        raise NotImplementedError()

    @abstractmethod
    def is_class_model(self) -> bool:
        raise NotImplementedError()

    def fit_and_predict(self, x_train, y_train, x_test) -> tuple[Sequence, Sequence]:
        predictor = self.fit(x_train, y_train)
        predictions_on_train = predictor.predict(x_train)
        predictions_on_test = predictor.predict(x_test)
        return predictions_on_train, predictions_on_test


class Classifier(Predictor, ABC):

    def is_class_predictor(self) -> bool:
        return True

    def score_concordance_index(self, x_test, y_test) -> float:
        raise IllegalStateError("Called object is of class " + str(self.__class__))


class ClassModel(Model):
    """Abstract class for models able to learn creating a predictor."""

    @abstractmethod
    def fit(self, x, y, sample_weight: Optional = None) -> Classifier:
        """ x is a list of anything, each element being a sample.
            y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call.
            Weights are optional. If they are provided by the model does not support them, they are ignored."""
        raise NotImplementedError()

    def is_class_model(self) -> bool:
        return True


class TunableClassModel(ABC):
    """A model that has tunable hyperparameters."""

    @abstractmethod
    def tune(self, hyperparameters) -> ClassModel:
        raise NotImplementedError()


class SklearnPredictorWrapper(Predictor, ABC):

    def __init__(self, sklearn_predictor):
        self.__sklearn_predictor = sklearn_predictor

    def _sklearn_predictor(self):
        return self.__sklearn_predictor

    def predict(self, x):
        return self._sklearn_predictor().predict(x)

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()

    def score_concordance_index(self, x_test, y_test) -> float:
        raise IllegalStateError()

    def __str__(self) -> str:
        res = "Wrapper for SKLearn predictor " + str(self._sklearn_predictor())
        if isinstance(self.__sklearn_predictor, Pipeline):
            res += "\n" + str(self.__sklearn_predictor.steps[-1][1]) + "\n"
        return res


class InputTransformer(Named, ABC):

    @abstractmethod
    def apply(self, x):
        raise NotImplementedError()


class PredictorWithInputTransformer(Predictor, ABC):
    __inner: Predictor
    __transformer: InputTransformer

    def __init__(self, predictor, input_transformer: InputTransformer):
        self.__predictor = predictor
        self.__transformer = input_transformer

    def predict(self, x):
        x = self.__transformer.apply(x)
        return self.__inner.predict(x)

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        x = self.__transformer.apply(x)
        return self.__inner.predict_survival_probabilities(x, times)

    def score_concordance_index(self, x_test, y_test) -> float:
        x_test = self.__transformer.apply(x_test)
        return self.__inner.score_concordance_index(x_test, y_test)

    def __str__(self) -> str:
        return ("Predictor with input transformer\n" +
                name_value("input transformer", self.__transformer) + "\n" +
                name_value("predictor", self.__predictor) + "\n")


class SklearnClassifierWrapper(SklearnPredictorWrapper, Classifier):

    def __init__(self, sklearn_predictor):
        SklearnPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor)


class SKLearnModelFactory(ABC):

    @abstractmethod
    def create(self):
        raise NotImplementedError()

    @abstractmethod
    def supports_weights(self) -> bool:
        raise NotImplementedError()
