from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor, SVtoMVPredictorWrapper
from model.sv_model import SampleWeight, SVModel, SVPredictor
from util.dataframe.dataframes import has_non_finite_error
from util.named import NickNamed
from util.table.backed_table import BackedTable
from util.table.table_backend.np_table import NpTable
from views.views import Views, JustViews


class MVModel(NickNamed):
    """Abstract class for multi-view models able to learn creating a predictor."""

    @abstractmethod
    def fit(self, data: ModelReadyInputData, sample_weight: Optional[SampleWeight] = None) -> MVPredictor:
        """ Returns a Predictor. The model itself is not affected by the call.
            We pass a simplified kind of InputData, to guarantee that it is well-formed.
            Weights are optional. If they are provided but the model does not support them, they are ignored."""
        raise NotImplementedError()

    def checked_fit(self,
                    data: ModelReadyInputData,
                    sample_weight: Optional[SampleWeight] = None,
                    check_non_finite: bool = False) -> MVPredictor:
        """This method checks for non-finite values in the views, then calls fit."""
        if data is None:
            raise ValueError("data is None")
        if check_non_finite:
            for view in data.views():
                if view.has_non_finite():
                    raise has_non_finite_error(df=view.to_dataframe())
        return self.fit(data=data, sample_weight=sample_weight)

    def fit_and_predict(
            self, train_data: ModelReadyInputData, views_test: Views,
            train_sample_weight: Optional[SampleWeight] = None) -> tuple[Sequence,Sequence]:
        predictor = self.fit(data=train_data, sample_weight=train_sample_weight)
        predictions_on_train = predictor.predict(views=train_data.views())
        predictions_on_test = predictor.predict(views=views_test)
        return predictions_on_train, predictions_on_test

    def as_sv_model(self) -> SVModel:
        return MvToSvModelAdapter(mv_model=self)


class MvToSvModelAdapter(SVModel):
    __mv_model: MVModel

    def __init__(self, mv_model: MVModel):
        self.__mv_model = mv_model

    def fit(self, x: DataFrame, y, sample_weight: Optional[SampleWeight] = None) -> SVPredictor:
        data = ModelReadyInputData.create_raw(x=x, y=y)
        mv_predictor = self.__mv_model.fit(data=data, sample_weight=sample_weight)
        return MvToSvPredictorAdapter(mv_predictor=mv_predictor)

    def nick(self) -> str:
        return self.__mv_model.nick()


class MvToSvPredictorAdapter(SVPredictor):
    __mv_predictor: MVPredictor

    def __init__(self, mv_predictor: MVPredictor):
        self.__mv_predictor = mv_predictor

    def predict(self, x: DataFrame) -> Sequence:
        table = BackedTable(NpTable(data=x))
        return self.__mv_predictor.predict(views=JustViews(views_dict={"x": table}))

    def predict_crisp(self, x: DataFrame) -> Sequence:
        table = BackedTable(NpTable(data=x))
        return self.__mv_predictor.predict_crisp(views=JustViews(views_dict={"x": table}))

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        return self.__mv_predictor.score_concordance_index(data = ModelReadyInputData.create_raw(x=x_test, y=y_test))

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        table = BackedTable(NpTable(data=x))
        return self.__mv_predictor.predict_survival_probabilities(views=JustViews(views_dict={"x": table}), times=times)


class SvToMvModelWrapper(MVModel):
    """The views are collapsed before use."""
    __sv_model: SVModel

    def __init__(self, sv_model: SVModel):
        self.__sv_model = sv_model

    def fit(self, data: ModelReadyInputData, sample_weight: Optional[SampleWeight] = None) -> MVPredictor:
        if data.needs_adjustment():
            raise ValueError("Data should be already adjusted at this stage.")
        sv_predictor = self.__sv_model.fit(
            x=data.collapsed_views().to_dataframe(), y=data.outcome_data(), sample_weight=sample_weight)
        return SVtoMVPredictorWrapper(sv_predictor=sv_predictor)

    def nick(self) -> str:
        return self.__sv_model.nick()

    def name(self) -> str:
        return self.__sv_model.name()

    def __str__(self) -> str:
        return str(self.__sv_model)
