from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame

from input_data.model_ready_input_data import ModelReadyInputData
from model.sv_model import SVPredictor, SampleWeight, SVModel
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from util.list_like import ListLike
from views.views import Views


class MaskedMVPredictor(MVPredictor):
    """Input is collapsed and then masked."""
    __mask: ListLike  # Mask is applied on collapsed views.
    __inner_predictor: SVPredictor  # A single-view predictor.

    def __init__(self, mask: ListLike, inner_predictor: SVPredictor):
        self.__mask = mask
        self.__inner_predictor = inner_predictor

    def __collapse_views_and_filter_by_mask(self, views: Views) -> DataFrame:
        return views.collapsed_filtered_by_mask(mask=self.__mask).to_dataframe()

    def predict(self, views: Views) -> Sequence:
        collapsed_x = self.__collapse_views_and_filter_by_mask(views=views)
        try:
            return self.__inner_predictor.predict(x=collapsed_x)
        except Exception as e:
            print("Exception while calling inner predict")
            print("inner predictor type: " + str(type(self.__inner_predictor)))
            print("inner predictor: " + str(self.__inner_predictor))
            raise e

    def predict_crisp(self, views: Views) -> Sequence:
        collapsed_x = self.__collapse_views_and_filter_by_mask(views=views)
        try:
            return self.__inner_predictor.predict_crisp(x=collapsed_x)
        except Exception as e:
            print("Exception while calling inner predict")
            print("inner predictor type: " + str(type(self.__inner_predictor)))
            print("inner predictor: " + str(self.__inner_predictor))
            raise e

    def predict_survival_probabilities(self, x: Views, times: Sequence[float]) -> DataFrame:
        """Return probabilities that event has not happened up to the passed times.
        It returns times on the rows and individuals on the columns."""
        x = self.__collapse_views_and_filter_by_mask(views=x)
        return self.__inner_predictor.predict_survival_probabilities(x=x, times=times)

    def score_concordance_index(self, test_data: ModelReadyInputData) -> float:
        x = self.__collapse_views_and_filter_by_mask(views=test_data.views())
        try:
            return self.__inner_predictor.score_concordance_index(x_test=x, y_test=test_data.outcome_data())
        except Exception as e:
            print("Exception while calling inner score_concordance_index")
            print("inner predictor type: " + str(type(self.__inner_predictor)))
            print("inner predictor: " + str(self.__inner_predictor))
            raise e

    def __str__(self) -> str:
        res = "Masked multi-view predictor\n"
        res += "Mask:\n"
        res += str(self.__mask) + "\n"
        res += "Inner predictor: " + str(self.__inner_predictor) + "\n"
        return res


class MaskedMVModel(MVModel):
    """If data needs adjustment, it is ignored and all the features are used as directly predictive."""
    __mask: ListLike  # Mask is applied on collapsed views.
    __inner_model: SVModel

    def __init__(self, mask: ListLike, model: SVModel):
        self.__mask = mask
        self.__model = model

    def fit(self, data: ModelReadyInputData, sample_weight: Optional[SampleWeight] = None) -> MVPredictor:
        x = data.views().collapsed_filtered_by_mask(mask=self.__mask)
        try:
            inner_predictor = self.__model.fit(x=x.to_dataframe(), y=data.outcome_data(), sample_weight=sample_weight)
        except Exception as e:
            print("Exception while fitting inner predictor")
            print("inner model type: " + str(type(self.__model)))
            print("inner model: " + str(self.__model))
            raise e
        return MaskedMVPredictor(mask=self.__mask, inner_predictor=inner_predictor)

    def nick(self) -> str:
        res = "Masked multi-view model\n"
        res += "Mask:\n"
        res += str(self.__mask) + "\n"
        res += "Inner model: " + str(self.__inner_model) + "\n"
        return res
