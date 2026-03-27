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
from util.table.table_utils import n_col
from util.utils import IllegalStateError


DEFAULT_RESET_OUTCOME_INDEX = False


def format_y(y, reset_index: bool = DEFAULT_RESET_OUTCOME_INDEX) -> DataFrame:
    """Formatting only dataframes at the moment. Other structures are left unchanged, but it attempts to convert to
    DataFrame at the beginning.
    Dataframes with both 2 or more columns and 2 or more rows are left unchanged.
    The resulting dataframe is guaranteed to have an index made of strings."""
    if not isinstance(y, pd.DataFrame):
        y = DataFrame(y)
    if reset_index:
        y = y.reset_index(drop=True)
    y.index = y.index.astype(str)
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

    def __init__(self, data: DataFrame, descriptor: OutcomeDescriptor, reset_index: bool = DEFAULT_RESET_OUTCOME_INDEX):
        self.__data = format_y(data, reset_index=reset_index)
        if self.__data.isnull().values.any():
            raise ValueError("NaNs present in outcome data.")
        self.__descriptor = descriptor

    def name(self) -> str:
        return self.__descriptor.name()

    def type(self) -> OutcomeType:
        return self.__descriptor.outcome_type()

    def data(self) -> DataFrame:
        return self.__data

    def first_col(self) -> list:
        """Returns the first column as a list."""
        return self.data().iloc[:, 0].tolist()

    @abstractmethod
    def select_by_row_indices(self, indices: Sequence[int]) -> Outcome:
        raise NotImplementedError()

    def __str__(self) -> str:
        return str(self.__descriptor)

    @abstractmethod
    def class_labels(self) -> Sequence:
        raise NotImplementedError()

    def is_categorical(self) -> bool:
        return self.type() == OutcomeType.categorical

    def is_survival(self) -> bool:
        return self.type() == OutcomeType.survival

    @abstractmethod
    def is_binary(self) -> bool:
        raise NotImplementedError()

    def row_names(self) -> Sequence[str]:
        return self.__data.index


class CategoricalOutcome(Outcome):
    __labels: Sequence

    def __init__(self, data: DataFrame, name: str, labels: Optional[Sequence] = None,
                 reset_index: bool = DEFAULT_RESET_OUTCOME_INDEX):
        """labels are the labels of the classes. Will get inferred from the values if not passed."""
        Outcome.__init__(self, data=data, descriptor=OutcomeDescriptorCategorical(name=name), reset_index=reset_index)
        if labels is None:
            counter = collections.Counter(self.first_col()).most_common()
            self.__labels = [c[0] for c in counter]
        else:
            self.__labels = labels

    def class_labels(self) -> Sequence:
        """In order of decreasing frequency."""
        return self.__labels

    def select_by_row_indices(self, indices: Sequence[int]) -> CategoricalOutcome:
        res_data = self.data().iloc[indices]
        return CategoricalOutcome(data=res_data, name=self.name(), labels=self.__labels, reset_index=False)

    def is_binary(self) -> bool:
        return len(self.class_labels()) == 2


class SurvivalOutcome(Outcome):

    def __init__(self, data: DataFrame, name: str, reset_index: bool = DEFAULT_RESET_OUTCOME_INDEX):
        Outcome.__init__(self, data=data, descriptor=OutcomeDescriptorSurvival(name=name), reset_index=reset_index)

    def class_labels(self) -> Sequence:
        raise IllegalStateError()

    def select_by_row_indices(self, indices: Sequence[int]) -> SurvivalOutcome:
        res_data = self.data().iloc[indices]
        return SurvivalOutcome(data=res_data, name=self.name(), reset_index=False)

    def is_binary(self) -> bool:
        return False


def smart_create_outcome(y, name: str = "unnamed") -> Outcome:
    """Converts to DataFrame, then if there is one column assumes categorical,
    and if there are 2 columns assumes survival."""
    data = format_y(y)
    num_cols = n_col(data)
    if num_cols == 1:
        # Assuming categorical
        return CategoricalOutcome(data=data, name=name)
    elif num_cols == 2:
        # Assuming survival
        return SurvivalOutcome(data=data, name=name)
    else:
        raise ValueError()