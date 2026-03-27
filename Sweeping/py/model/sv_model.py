from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterable
from typing import Optional, overload, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from util.named import NickNamed, Named
from util.utils import IllegalStateError
from util.str_utils import name_value

DUMMY_STR = "dummy"


SampleWeight = Union[
    Sequence[float],           # lists/tuples of floats
    Sequence[int],             # lists/tuples of ints (allowed by many estimators)
    NDArray[np.number],        # numpy arrays (any numeric dtype)
    pd.Series,                 # pandas Series
]


class Predictor(ABC):
    pass


class SVPredictor(Predictor, ABC):

    @abstractmethod
    def predict(self, x: DataFrame) -> Sequence:
        raise NotImplementedError()

    @abstractmethod
    def predict_crisp(self, x: DataFrame) -> Sequence:
        raise NotImplementedError()

    @abstractmethod
    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        raise NotImplementedError()

    @abstractmethod
    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        """Return probabilities that event has not happened up to the passed times.
        It returns times on the rows and individuals on the columns."""
        raise NotImplementedError()


class SVModel(NickNamed):

    @abstractmethod
    def fit(self, x: DataFrame, y, sample_weight: Optional[SampleWeight] = None) -> SVPredictor:
        """ y is a list of anything, each element being an expected output.
            Returns a Predictor. The model itself is not affected by the call.
            Weights are optional. If they are provided by the model does not support them, they are ignored."""
        raise NotImplementedError()

    def fit_and_predict(self, x_train: DataFrame, y_train, x_test: DataFrame) -> tuple[Sequence, Sequence]:
        predictor = self.fit(x_train, y_train)
        predictions_on_train = predictor.predict(x_train)
        predictions_on_test = predictor.predict(x_test)
        return predictions_on_train, predictions_on_test


class PredictStrategy(ABC):

    @abstractmethod
    def predict(self, sklearn_predictor, x: DataFrame):
        raise NotImplementedError()


class JustPredictStrategy(PredictStrategy):

    def predict(self, sklearn_predictor, x: DataFrame):
        return sklearn_predictor.predict(x)


class PredictProbaResult(Sequence[Sequence[float]]):
    __classes: Sequence[str]
    __probabilities: np.ndarray
    """External sequence is for samples, internal sequences are for class probabilities.
    Internally an array is used for performance reasons."""

    def __init__(self, classes: Sequence[str], probabilities: Union[np.ndarray, Sequence[Sequence[float]]]):
        self.__classes = classes
        if not isinstance( probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        self.__probabilities = probabilities

    def classes(self) -> Sequence[str]:
        return self.__classes

    def probabilities(self) -> np.ndarray:
        return self.__probabilities

    @overload
    def __getitem__(self, index: int) -> Sequence[float]: ...

    @overload
    def __getitem__(self, index: slice) -> PredictProbaResult: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PredictProbaResult(classes=self.classes(), probabilities=self.__probabilities[index])
        else:
            return self.__probabilities[index]

    def __len__(self) -> int:
        return len(self.__probabilities)

    def select_by_indices(self, indices: Iterable[int]) -> PredictProbaResult:
        proba = self.__probabilities
        return PredictProbaResult(classes=self.classes(), probabilities=[proba[i] for i in indices])

    def select_by_class_position(self, class_position: int) -> np.ndarray:
        """Returns a 1D array with shape (n_rows,)."""
        return self.__probabilities[:, class_position]

    def as_df(self) -> DataFrame:
        return pd.DataFrame(self.__probabilities, columns=list(self.__classes))

    def __str__(self) -> str:
        return str(self.as_df())


def clean_probabilities(probs: NDArray, strategy: str = "normalize"):
    """
    Cleans in place NaN or Inf values from a predict_proba output.

    Parameters:
    - probs: np.ndarray, shape (n_samples, n_classes)
    - strategy: str, one of ["uniform", "normalize"]
        - "uniform": replace invalid rows with uniform probabilities
        - "normalize": replace invalid values with 0 and renormalize the row
    """

    # Identify rows with any non-finite values
    try:
        invalid_mask = ~np.isfinite(probs).all(axis=1)
    except TypeError as e:
        print("The input probs should be a np.ndarray with shape (n_samples, n_classes).")
        raise e

    for i in np.where(invalid_mask)[0]:
        if strategy == "uniform":
            probs[i] = np.full(probs.shape[1], 1.0 / probs.shape[1])
        elif strategy == "normalize":
            probs[i][~np.isfinite(probs[i])] = 0.0
            total = probs[i].sum()
            probs[i] = probs[i] / total if total > 0 else np.full(probs.shape[1], 1.0 / probs.shape[1])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class PredictProbaStrategy(PredictStrategy):

    def predict(self, sklearn_predictor, x: DataFrame) -> PredictProbaResult:
        probs = sklearn_predictor.predict_proba(x)
        clean_probabilities(probs=probs)
        return PredictProbaResult(
            classes=list(sklearn_predictor.classes_), probabilities=probs)


class SklearnSVPredictorWrapper(SVPredictor, ABC):
    __predict_strategy: PredictStrategy

    def __init__(self, sklearn_predictor, predict_strategy: PredictStrategy = JustPredictStrategy()):
        self.__sklearn_predictor = sklearn_predictor
        self.__predict_strategy = predict_strategy

    def _get_sklearn_predictor(self):
        return self.__sklearn_predictor

    def predict(self, x: DataFrame):
        return self.__predict_strategy.predict(sklearn_predictor=self._get_sklearn_predictor(), x=x)

    def predict_crisp(self, x: DataFrame) -> Sequence:
        return JustPredictStrategy().predict(sklearn_predictor=self._get_sklearn_predictor(), x=x)

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise IllegalStateError()

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        raise IllegalStateError()

    def __str__(self) -> str:
        res = "Wrapper for SKLearn predictor " + str(self._get_sklearn_predictor())
        if isinstance(self.__sklearn_predictor, Pipeline):
            res += "\n" + str(self.__sklearn_predictor.steps[-1][1]) + "\n"
        return res


class InputTransformer(Named, ABC):

    @abstractmethod
    def apply(self, x: DataFrame) -> DataFrame:
        raise NotImplementedError()


class SVPredictorWithInputTransformer(SVPredictor, ABC):
    __inner: SVPredictor
    __transformer: InputTransformer

    def __init__(self, predictor: SVPredictor, input_transformer: InputTransformer):
        self.__predictor = predictor
        self.__transformer = input_transformer

    def predict(self, x: DataFrame):
        x = self.__transformer.apply(x)
        return self.__inner.predict(x)

    def predict_crisp(self, x: DataFrame):
        x = self.__transformer.apply(x)
        return self.__inner.predict_crisp(x)

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        x = self.__transformer.apply(x)
        return self.__inner.predict_survival_probabilities(x, times)

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        x_test = self.__transformer.apply(x_test)
        return self.__inner.score_concordance_index(x_test, y_test)

    def __str__(self) -> str:
        return ("Predictor with input transformer\n" +
                name_value("input transformer", self.__transformer) + "\n" +
                name_value("predictor", self.__predictor) + "\n")


class SKLearnModelFactory(NickNamed, ABC):

    @abstractmethod
    def create(self):
        raise NotImplementedError()

    @abstractmethod
    def supports_weights(self) -> bool:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name()
