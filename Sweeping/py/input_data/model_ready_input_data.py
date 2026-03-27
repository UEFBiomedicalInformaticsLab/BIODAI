from __future__ import annotations
from typing import Union, Optional, Sequence

from pandas import DataFrame

from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from input_data.input_data import InputData
from input_data.outcome import Outcome, smart_create_outcome
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews, EmptyViews


class ModelReadyInputData(EvaluationReadyInputData):
    __outcome: Outcome

    def __init__(self, all_views: Union[dict[str, Union[DataFrame, Table]], Views], outcome: Outcome,
                 nick: str, adjusted_views: Optional[AdjustedViewDef] = None, name: Optional[str] = None):
        """Assuming all views have the same sample at the same row.
        Constructor checks if the views have the same number of samples.
        Stratify outcome is considered to be the one outcome that is present.
        There are no covariate views."""
        if isinstance(all_views, dict) and len(all_views) == 0:
            all_views = EmptyViews(sample_names=outcome.row_names())
        InputData.__init__(self=self, all_views=all_views, nick=nick, adjusted_views=adjusted_views, name=name)
        self.__outcome = outcome
        if not self._n_samples_consistency():
            raise ValueError("Number of samples is not consistent.\n" + str(self))

    def outcomes(self) -> Sequence[Outcome]:
        return [self.__outcome]

    def has_stratify_outcome(self) -> bool:
        return True

    def stratify_outcome_name(self) -> str:
        return self.__outcome.name()

    def outcome_data(self) -> DataFrame:
        """Returns the data from the only outcome that is present."""
        return self.__outcome.data()

    def model_ready(self, outcome: Optional[str] = None) -> ModelReadyInputData:
        if outcome is None or outcome == self.__outcome.name():
            return self
        else:
            raise ValueError()

    @staticmethod
    def create_raw(x: Union[Views, DataFrame, Table], y, nick: str = "data", name: Optional[str] = None
                   ) -> ModelReadyInputData:
        """Index of dataframes is not reset."""
        if isinstance(x, DataFrame):
            x = BackedTable(backend=NpTable(data=x))
        if isinstance(x, Table):
            x = JustViews(views_dict={"x": x})
        return ModelReadyInputData(all_views=x, outcome=smart_create_outcome(y), nick=nick, name=name)
