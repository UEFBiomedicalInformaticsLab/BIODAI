from abc import abstractmethod, ABC
from collections.abc import Sequence

import numpy as np
import pandas as pd

from util.math.online_variance_builder import OnlineVarianceBuilder


class FittedFeatureStandardScaler:
    __mean: float
    __std: float

    def __init__(self, mean: float, std: float):
        self.__mean = mean
        self.__std = std

    def transform(self, elements: Sequence[float]) -> np.ndarray:
        if pd.isna(self.__mean):
            return np.array(elements)
        else:
            if pd.isna(self.__std) or self.__std == 0.0:
                return np.array(elements) - self.__mean
            else:
                return (np.array(elements) - self.__mean) / self.__std


class FeatureStandardScaler(ABC):
    """A faster alternative to StandardScaler for scaling only one feature."""
    @abstractmethod
    def fit(self, data: Sequence[float]) -> FittedFeatureStandardScaler:
        raise NotImplementedError()



class FeatureStandardScalerKahan(FeatureStandardScaler):

    def fit(self, data: Sequence[float]) -> FittedFeatureStandardScaler:
        builder = OnlineVarianceBuilder()
        builder.add_all(data)
        return FittedFeatureStandardScaler(mean=builder.mean(), std=builder.biased_standard_deviation())


class FeatureStandardScalerNumpy(FeatureStandardScaler):
    """Less precise but a lot faster than Kahan version."""

    def fit(self, data: Sequence[float]) -> FittedFeatureStandardScaler:
        return FittedFeatureStandardScaler(mean=np.mean(data), std=np.std(data))
