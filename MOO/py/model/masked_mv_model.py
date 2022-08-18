from pandas import DataFrame

from model.model import ClassModel, Predictor
from model.multi_view_model import MVModel, MVPredictor
from multi_view_utils import collapse_views_and_filter_by_mask
from views.views import Views


class MaskedMVPredictor(MVPredictor):

    def __init__(self, mask, inner_predictor: Predictor):
        self.__mask = mask
        self.__inner_predictor = inner_predictor

    def __collapse_views_and_filter_by_mask(self, views) -> DataFrame:
        """TODO Temporary code while all views implement Views."""
        if isinstance(views, Views):
            return views.collapsed_filtered_by_mask(mask=self.__mask)
        else:
            return collapse_views_and_filter_by_mask(views=views, mask=self.__mask)

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

    def score_concordance_index(self, x_test, y_test) -> float:
        """x_test is a dict of views."""
        x = self.__collapse_views_and_filter_by_mask(views=x_test)
        try:
            return self.__inner_predictor.score_concordance_index(x_test=x, y_test=y_test)
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
