from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from numpy import number

from descriptor.descriptor import Described, Descriptor
from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from util.table.table import Table


class UnivariatePropertyComputer(Described, ABC):

    def __init__(self, descriptor: Optional[Descriptor] = None):
        Described.__init__(self=self, descriptor=descriptor)

    @abstractmethod
    def outcome_types(self) -> Sequence[OutcomeType]:
        """Supported outcome types."""
        raise NotImplementedError()

    @abstractmethod
    def inner_compute_property(self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None):
        raise NotImplementedError()

    def compute_property(self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None):
        """Covariates are additional variables that might be taken into account,
        depending on the specific property computer."""
        if outcome.type() not in self.outcome_types():
            raise ValueError("Input outcome type does not match this property computer.")
        return self.inner_compute_property(feature=feature, outcome=outcome, covariates=covariates)
