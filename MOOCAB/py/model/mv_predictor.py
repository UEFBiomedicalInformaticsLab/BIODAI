from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from pandas import DataFrame

from model.model import Classifier
from multi_view_utils import collapse_views
from util.feature_space_lifter import FeatureSpaceLifterMV
from views.views import Views


class MVPredictor(ABC):

    # x is a dict of views, each being a matrix with samples on rows.
    # Returns a list of anything, each element being an expected output.
    @abstractmethod
    def predict(self, views: dict[str, DataFrame]):
        raise NotImplementedError()

    @abstractmethod
    def score_concordance_index(self, views: dict[str, DataFrame], y) -> float:
        raise NotImplementedError()

    def downlift(self, lifter: FeatureSpaceLifterMV) -> MVPredictor:
        return DownliftedMVPredictor(inner_predictor=self, lifter=lifter)


class SVtoMVPredictorWrapper(MVPredictor):

    __inner: Classifier

    def __init__(self, sv_predictor: Classifier):
        self.__inner = sv_predictor

    # x is a list of views, each being a matrix with samples on rows.
    # Returns a list of anything, each element being an expected output.
    def predict(self, views: dict[str, DataFrame]):
        collapsed_views = collapse_views(views)
        return self.__inner.predict(x=collapsed_views)

    def score_concordance_index(self, views: dict[str, DataFrame], y) -> float:
        collapsed_views = collapse_views(views)
        return self.__inner.score_concordance_index(x_test=collapsed_views, y_test=y)

    def __str__(self):
        return "Single view to multi view predictor wrapper with inner predictor " + str(self.__inner)


class DownliftedMVPredictor(MVPredictor):
    __inner: MVPredictor
    __lifter: FeatureSpaceLifterMV

    def __init__(self, inner_predictor: MVPredictor, lifter: FeatureSpaceLifterMV):
        self.__inner = inner_predictor
        self.__lifter = lifter

    def predict(self, views: dict[str, DataFrame]):
        return self.__inner.predict(views=self.__lifter.uplift_dict(views))

    def score_concordance_index(self, views: Union[dict[str, DataFrame], Views], y) -> float:
        return self.__inner.score_concordance_index(self.__lifter.uplift_dict(views), y)

    def __str__(self):
        return "Downlifted multi view predictor with inner predictor " + str(self.__inner)
