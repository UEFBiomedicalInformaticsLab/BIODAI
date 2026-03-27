from typing import Sequence

from pandas import DataFrame

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from view_adjuster.views_adjuster import ViewsAdjuster
from views.views import Views


class AdjustedMVPredictor(MVPredictor):
    __views_adjuster: ViewsAdjuster
    __inner_predictor: MVPredictor

    def __init__(self, views_adjuster: ViewsAdjuster, inner_predictor: MVPredictor):
        self.__views_adjuster = views_adjuster
        self.__inner_predictor = inner_predictor

    def predict(self, views: Views) -> Sequence:
        views = self.__views_adjuster.adjust(views)
        return self.__inner_predictor.predict(views)

    def predict_crisp(self, views: Views) -> Sequence:
        views = self.__views_adjuster.adjust(views)
        return self.__inner_predictor.predict_crisp(views)

    def score_concordance_index(self, data: ModelReadyInputData) -> float:
        views = self.__views_adjuster.adjust(data.views())
        data = data.set_views(views)
        return self.__inner_predictor.score_concordance_index(data)

    def predict_survival_probabilities(self, views: Views, times: Sequence[float]) -> DataFrame:
        views = self.__views_adjuster.adjust(views)
        return self.__inner_predictor.predict_survival_probabilities(views=views, times=times)
