from __future__ import annotations

from abc import ABC
from typing import Sequence, Union, Optional

from pandas import DataFrame

from input_data.input_data import InputData
from input_data.outcome import Outcome, smart_create_outcome
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.utils import IllegalStateError
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews


class EvaluationReadyInputData(InputData, ABC):
    """Either has one outcome or zero outcomes, and no covariate views."""

    def covariate_view_names(self) -> Sequence[str]:
        return []

    def the_outcome(self) -> Outcome:
        return self.outcomes()[0]

    @staticmethod
    def create_raw(x: Union[Views, DataFrame, Table], y, nick: str = "data", name: Optional[str] = None
                   ) -> EvaluationReadyInputData:
        """Index of dataframes is not reset."""
        if isinstance(x, DataFrame):
            x = BackedTable(backend=NpTable(data=x))
        if isinstance(x, Table):
            x = JustViews(views_dict={"x": x})
        if y is None:
            return NoOutcomesInputData(all_views=x, nick=nick, name=name)
        else:
            from input_data.model_ready_input_data import ModelReadyInputData
            return ModelReadyInputData(all_views=x, outcome=smart_create_outcome(y), nick=nick, name=name)


class NoOutcomesInputData(EvaluationReadyInputData):
    def __init__(
            self, all_views: Union[dict[str, Union[DataFrame, Table]], Views],
            nick: str, adjusted_views: Optional[AdjustedViewDef] = None, name: Optional[str] = None):
        """Assuming all views have the same sample at the same row.
                Constructor checks if the views have the same number of samples.
                Stratify outcome is considered to be the one outcome that is present.
                There are no covariate views."""
        InputData.__init__(self=self, all_views=all_views, nick=nick, adjusted_views=adjusted_views, name=name)

    def outcomes(self) -> Sequence[Outcome]:
        return []

    def has_stratify_outcome(self) -> bool:
        return False

    def stratify_outcome_name(self) -> str:
        raise IllegalStateError()
