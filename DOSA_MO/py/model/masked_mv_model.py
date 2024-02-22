from collections.abc import Sequence
from typing import Union

from pandas import DataFrame

from model.model import ClassModel, Predictor
from model.multi_view_model import MVModel, MVTunableModel
from model.mv_predictor import MVPredictor
from multi_view_utils import collapse_views_and_filter_by_mask, collapse_iterable_dfs_and_filter_by_mask
from util.list_like import ListLike
from views.views import Views


class MaskedMVPredictor(MVPredictor):
    """Input is collapsed and then masked."""
    __mask: ListLike  # Mask is applied on collapsed views.
    __inner_predictor: Predictor  # A single-view predictor.

    def __init__(self, mask, inner_predictor: Predictor):
        self.__mask = mask
        self.__inner_predictor = inner_predictor

    def __collapse_views_and_filter_by_mask(self, views) -> DataFrame:
        """TODO Temporary code while all views implement Views."""
        if isinstance(views, Views):
            return views.collapsed_filtered_by_mask(mask=self.__mask)
        elif isinstance(views, dict):
            return collapse_views_and_filter_by_mask(views=views, mask=self.__mask)
        else:  # Assuming iterable
            return collapse_iterable_dfs_and_filter_by_mask(dfs=views, mask=self.__mask)

    def predict(self, views):
        """x is a dict of views."""
        collapsed_x = self.__collapse_views_and_filter_by_mask(views=views)
        try:
            return self.__inner_predictor.predict(x=collapsed_x)
        except Exception as e:
            print("Exception while calling inner predict")
            print("inner predictor type: " + str(type(self.__inner_predictor)))
            print("inner predictor: " + str(self.__inner_predictor))
            raise e

    def score_concordance_index(self, x_test: Union[Views, dict[str, DataFrame]], y_test) -> float:
        """x_test is a Views object or a dict of views."""
        x = self.__collapse_views_and_filter_by_mask(views=x_test)
        try:
            return self.__inner_predictor.score_concordance_index(x_test=x, y_test=y_test)
        except Exception as e:
            print("Exception while calling inner score_concordance_index")
            print("inner predictor type: " + str(type(self.__inner_predictor)))
            print("inner predictor: " + str(self.__inner_predictor))
            raise e

    def predict_survival_probabilities(self, x, times: Sequence[float]) -> DataFrame:
        """Return probabilities that event has not happened up to the passed times.
        It returns times on the rows and individuals on the columns."""
        x = self.__collapse_views_and_filter_by_mask(views=x)
        return self.__inner_predictor.predict_survival_probabilities(x=x, times=times)

    def __str__(self) -> str:
        res = "Masked multi-view predictor\n"
        res += "Mask:\n"
        res += str(self.__mask) + "\n"
        res += "Inner predictor: " + str(self.__inner_predictor) + "\n"
        return res


class MaskedMVModel(MVModel):

    def __init__(self, mask, model: ClassModel):
        self.__mask = mask
        self.__model = model

    def fit(self, views, y) -> MVPredictor:
        x = collapse_views_and_filter_by_mask(views=views, mask=self.__mask)
        try:
            inner_predictor = self.__model.fit(x=x, y=y)
        except Exception as e:
            print("Exception while fitting inner predictor")
            print("inner model type: " + str(type(self.__model)))
            print("inner model: " + str(self.__model))
            raise e
        return MaskedMVPredictor(mask=self.__mask, inner_predictor=inner_predictor)


# Defines models with a mask: a list of 1 and 0 that selects features.
class MaskedMVTunableModel(MVTunableModel):

    def __init__(self, model: ClassModel):
        self.__model = model

    def tune(self, mask) -> MaskedMVModel:
        return MaskedMVModel(mask, self.__model)
