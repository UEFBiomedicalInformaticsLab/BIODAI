from typing import Optional

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.adjusted_mv_predictor import AdjustedMVPredictor
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from model.sv_model import SampleWeight
from view_adjuster.views_adjuster import ViewsAdjusterModel, DEFAULT_VIEWS_ADJUSTER_MODEL_NICK


class AdjustedMVModel(MVModel):
    __views_adjuster_model: ViewsAdjusterModel
    __inner_model: MVModel

    def __init__(self, views_adjuster_model: ViewsAdjusterModel, inner_model: MVModel):
        self.__views_adjuster_model = views_adjuster_model
        self.__inner_model = inner_model

    def fit(self, data: ModelReadyInputData, sample_weight: Optional[SampleWeight] = None) -> MVPredictor:
        views = data.views()
        views_adjuster = self.__views_adjuster_model.fit(
            views=views, adjusted_view_def=data.adjusted_view_def(), sample_weight=sample_weight)
        adjusted_data = views_adjuster.adjust_input_data(input_data=data)
        inner_predictor = self.__inner_model.fit(data=adjusted_data, sample_weight=sample_weight)
        return AdjustedMVPredictor(views_adjuster=views_adjuster, inner_predictor=inner_predictor)

    def nick(self) -> str:
        adjuster_nick = self.__views_adjuster_model.nick()
        if adjuster_nick == DEFAULT_VIEWS_ADJUSTER_MODEL_NICK:
            return self.__inner_model.nick()
        else:
            return adjuster_nick + "_" + self.__inner_model.nick()

    def name(self) -> str:
        return self.__views_adjuster_model.name() + "_" + self.__inner_model.name()

    def __str__(self) -> str:
        return str(self.__views_adjuster_model) + " -> " + str(self.__inner_model)