from collections.abc import Iterable
from typing import Optional, Sequence, Union

from pandas import DataFrame

from input_data.input_data import InputData
from input_data.outcome import Outcome
from util.table.table import Table
from util.utils import IllegalStateError
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views


class FullInputData(InputData):
    __outcomes: dict[str, Outcome]
    __stratify_outcome: Optional[str]
    __covariate_views: Sequence[str]
    """Covariate views are in alphabetical order."""

    def __init__(self, all_views: Union[dict[str, Union[DataFrame, Table]], Views], outcomes: Sequence[Outcome],
                 nick: str, stratify_outcome: Optional[str] = None, covariate_views: Optional[Iterable[str]] = None,
                 adjusted_views: Optional[AdjustedViewDef] = None, name: Optional[str] = None):
        """Assuming all views have the same sample at the same row.
        Constructor checks if the views have the same number of samples.
        Keeps original order of the outcomes."""
        InputData.__init__(self=self, all_views=all_views, nick=nick, adjusted_views=adjusted_views, name=name)
        self.__outcomes = {}
        for o in outcomes:
            self.__outcomes[o.name()] = o
        self.__stratify_outcome = stratify_outcome
        if not self._n_samples_consistency():
            raise ValueError("Number of samples is not consistent.\n" + str(self))
        if covariate_views is None:
            covariate_views = {}
        self.__covariate_views = sorted(set(covariate_views))
        for v in self.__covariate_views:
            if not v in self.predictive_view_names():
                raise ValueError("Trying to set a non predictive view as covariate.")

    def outcomes(self) -> Sequence[Outcome]:
        """Keeps original order."""
        return list(self.__outcomes.values())

    def has_stratify_outcome(self) -> bool:
        return self.__stratify_outcome is not None

    def stratify_outcome_name(self) -> str:
        """Default outcome on which to stratify."""
        if self.__stratify_outcome is not None:
            return self.__stratify_outcome
        else:
            raise IllegalStateError()

    def covariate_view_names(self) -> Sequence[str]:
        return self.__covariate_views
