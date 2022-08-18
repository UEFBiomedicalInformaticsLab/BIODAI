from abc import ABC, abstractmethod
from collections.abc import Sequence

from util.named import NickNamed
from util.utils import IllegalStateError


DEFAULT_LOGISTIC_MAX_ITER = 3000


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


class Model(NickNamed):

    @abstractmethod
    def fit(self, x, y) -> Predictor:
        """ x is a list of anything, each element being a sample.
            y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call."""
        raise NotImplementedError()

    @abstractmethod
    def is_class_model(self) -> bool:
        raise NotImplementedError()


class Classifier(Predictor, ABC):

    def is_class_predictor(self) -> bool:
        return True

    def score_concordance_index(self, x_test, y_test) -> float:
        raise IllegalStateError()


class ClassModel(Model):
    """Abstract class for models able to learn creating a predictor."""

    @abstractmethod
    def fit(self, x, y) -> Classifier:
        """ x is a list of anything, each element being a sample.
            y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call."""
        raise NotImplementedError()

    def fit_and_predict(self, x_train, y_train, x_test):
        predictor = self.fit(x_train, y_train)
        predictions_on_train = predictor.predict(x_train)
        predictions_on_test = predictor.predict(x_test)
        return predictions_on_train, predictions_on_test

    def is_class_model(self) -> bool:
        return True


class SklearnPredictorWrapper(Classifier):

    def __init__(self, sklearn_predictor):
        self.__sklearn_predictor = sklearn_predictor

    def _sklearn_predictor(self):
        return self.__sklearn_predictor

    def predict(self, x):
        return self._sklearn_predictor().predict(x)

    def __str__(self) -> str:
        return "Wrapper for SKLearn predictor " + str(self._sklearn_predictor())


class SKLearnModelFactory(ABC):

    @abstractmethod
    def create(self):
        raise NotImplementedError()
