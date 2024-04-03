from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional
import pandas as pd
from pandas import DataFrame

from input_data.outcome_descriptor import OutcomeDescriptor, OutcomeDescriptorCategorical, OutcomeDescriptorSurvival
from input_data.outcome_type import OutcomeType
from util.named import Named
from util.utils import IllegalStateError


def format_y(y) -> DataFrame:
    """Formatting only dataframes at the moment. Other structures are left unchanged, but it attempts to convert to
    DataFrame at the beginning.
    Dataframes with both 2 or more columns and 2 or more rows are left unchanged."""
    if not isinstance(y, pd.DataFrame):
        y = DataFrame(y)
    y = y.reset_index(drop=True)
    s = y.shape
    if s[0] == 1:
        return y.squeeze(axis=1)
    elif s[1] == 1:
        return y.squeeze(axis=0)
    elif s[0] == 0 and s[1] == 0:
        return DataFrame()
    else:
        return y


class Outcome(Named, ABC):
    __data: DataFrame
    __descriptor: OutcomeDescriptor

    def __init__(self, data: DataFrame, descriptor: OutcomeDescriptor):
        self.__data = format_y(data)
        if self.__data.isnull().values.any():
            raise ValueError("NaNs present in outcome data.")
        self.__descriptor = descriptor

    def name(self) -> str:
        return self.__descriptor.name()

    def type(self) -> OutcomeType:
        return self.__descriptor.outcome_type()

    def data(self) -> DataFrame:
        return self.__data

    def fist_col(self) -> list:
        return self.data().iloc[:, 0].tolist()

    @abstractmethod
    def select_by_row_indices(self, indices: [int]) -> Outcome:
        raise NotImplementedError()

    def __str__(self) -> str:
        return str(self.__descriptor)

    @abstractmethod
    def class_labels(self) -> Sequence:
        raise NotImplementedError()


class CategoricalOutcome(Outcome):
    __labels: Sequence

    def __init__(self, data: DataFrame, name: str, labels: Optional[Sequence] = None):
        Outcome.__init__(self, data=data, descriptor=OutcomeDescriptorCategorical(name=name))
        if labels is None:
            counter = collections.Counter(self.fist_col()).most_common()
            self.__labels = [c[0] for c in counter]
        else:
            self.__labels = labels

    def class_labels(self) -> Sequence:
        return self.__labels

    def select_by_row_indices(self, indices: [int]) -> CategoricalOutcome:
        res_data = self.data().iloc[indices]
        return CategoricalOutcome(data=res_data, name=self.name(), labels=self.__labels)


class SurvivalOutcome(Outcome):

    def __init__(self, data: DataFrame, name: str):
        Outcome.__init__(self, data=data, descriptor=OutcomeDescriptorSurvival(name=name))

    def class_labels(self) -> Sequence:
        raise IllegalStateError()

    def select_by_row_indices(self, indices: [int]) -> SurvivalOutcome:
        res_data = self.data().iloc[indices]
        return SurvivalOutcome(data=res_data, name=self.name())
