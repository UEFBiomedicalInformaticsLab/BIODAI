from abc import ABC, abstractmethod
from typing import Optional

from numpy import ndarray

from model.regression.regressors_library import Linear
from model.regression.svregressor import RegressorSVModel
from model.sv_model import SampleWeight
from util.named import NickNamed
from util.table.table import Table
from view_adjuster.view_adjuster import ViewAdjuster, FeatureAdjuster, IterativeViewAdjuster, FeatureAdjusterByRegressor
from views.views import Views


class ViewAdjusterModel(NickNamed, ABC):

    @abstractmethod
    def fit(self, view_to_adjust: Table, adjusting_views: Views,
            sample_weight: Optional[SampleWeight] = None) -> ViewAdjuster:
        raise NotImplementedError()

    def fit_adjust(self, view_to_adjust: Table, adjusting_views: Views,
                   sample_weight: Optional[SampleWeight] = None) -> Table:
        return self.fit(
            view_to_adjust=view_to_adjust, adjusting_views=adjusting_views, sample_weight=sample_weight).adjust(
            view_to_adjust=view_to_adjust, adjusting_views=adjusting_views)


class FeatureAdjusterModel(NickNamed, ABC):

    @abstractmethod
    def fit(self, feature_to_adjust: ndarray, adjusting_views: Views,
            sample_weight: Optional[SampleWeight] = None) -> FeatureAdjuster:
        """feature_to_adjust is a 1-dimensional array."""
        raise NotImplementedError()


class DummyViewAdjuster(ViewAdjuster):

    def adjust(self, view_to_adjust: Table, adjusting_views: Views) -> Table:
        return view_to_adjust


class IterativeViewAdjusterModel(ViewAdjusterModel):
    """Iterates on the features to be adjusted."""
    __feature_adjuster_model: FeatureAdjusterModel

    def __init__(self, feature_adjuster_model: FeatureAdjusterModel):
        self.__feature_adjuster_model = feature_adjuster_model

    def fit(self, view_to_adjust: Table, adjusting_views: Views,
            sample_weight: Optional[SampleWeight] = None) -> ViewAdjuster:
        if adjusting_views.n_views() == 0:
            return DummyViewAdjuster()
        adjusting_views = adjusting_views.as_cached()
        model = self.__feature_adjuster_model
        return IterativeViewAdjuster(
            feature_adjusters=(
                model.fit(feature_to_adjust=f, adjusting_views=adjusting_views, sample_weight=sample_weight)
                for f in view_to_adjust.np_cols()))

    def nick(self) -> str:
        return self.__feature_adjuster_model.nick()

    def name(self) -> str:
        return self.__feature_adjuster_model.name()

    def __str__(self) -> str:
        return "Iterative " + str(self.__feature_adjuster_model)


class FeatureAdjusterModelByRegressor(FeatureAdjusterModel):
    __model: RegressorSVModel

    def __init__(self, model: RegressorSVModel):
        self.__model = model

    def fit(self, feature_to_adjust: ndarray, adjusting_views: Views,
            sample_weight: Optional[SampleWeight] = None) -> FeatureAdjuster:
        return FeatureAdjusterByRegressor(
            regressor=self.__model.fit(
                x=adjusting_views.to_dataframe(), y=feature_to_adjust, sample_weight=sample_weight))

    def nick(self) -> str:
        return self.__model.nick()

    def name(self) -> str:
        return self.__model.name()

    def __str__(self) -> str:
        return str(self.__model) + " feature adjuster"


class LinearFeatureAdjusterModel(FeatureAdjusterModelByRegressor):

    def __init__(self):
        FeatureAdjusterModelByRegressor.__init__(self=self, model=Linear())


class LinearViewAdjusterModel(IterativeViewAdjusterModel):

    def __init__(self):
        IterativeViewAdjusterModel.__init__(self=self, feature_adjuster_model=LinearFeatureAdjusterModel())


DEFAULT_VIEW_ADJUSTER_MODEL = LinearViewAdjusterModel()
