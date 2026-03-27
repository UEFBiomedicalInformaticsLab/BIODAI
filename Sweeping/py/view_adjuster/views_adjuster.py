from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from input_data.model_ready_input_data import ModelReadyInputData
from model.sv_model import SampleWeight
from util.named import NickNamed
from view_adjuster.targeted_view_adjuster import TargetedViewAdjuster, TargetedViewAdjusterModel
from view_adjuster.view_adjuster_model import ViewAdjusterModel, DEFAULT_VIEW_ADJUSTER_MODEL
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews


class ViewsAdjuster(ABC):

    @abstractmethod
    def adjust(self, views: Views) -> Views:
        raise NotImplementedError()

    def adjust_input_data(self, input_data: ModelReadyInputData) -> ModelReadyInputData:
        """The views will be adjusted using this object.
        Data about views needing adjustment in the input data will be ignored.
        The output object will not report the need for any adjustment."""
        adjusted_views = self.adjust(views=input_data.views())
        return ModelReadyInputData(
            all_views=adjusted_views,
            outcome=input_data.the_outcome(),
            nick=input_data.nick())


class CompositeViewsAdjuster(ViewsAdjuster):
    __adjusters: Sequence[TargetedViewAdjuster]

    def __init__(self, adjusters: Sequence[TargetedViewAdjuster]):
        """All views part of the output must have an adjuster (potentially applying the identity function)."""
        self.__adjusters = adjusters

    def adjust(self, views: Views) -> Views:
        res_dict = {adjuster.adjusted_view_name(): adjuster.apply(views=views) for adjuster in self.__adjusters}
        return JustViews(views_dict=res_dict)


class ViewsAdjusterModel(NickNamed, ABC):

    @abstractmethod
    def fit(
            self, views: Views,
            adjusted_view_def: AdjustedViewDef,
            sample_weight: Optional[SampleWeight] = None) -> ViewsAdjuster:
        raise NotImplementedError()


class UniformViewsAdjusterModel(ViewsAdjusterModel):
    """It uses one kind of view adjuster model to do all adjustments."""
    __adjuster_model: ViewAdjusterModel

    def __init__(self, adjuster_model: ViewAdjusterModel):
        """It uses the passed kind of view adjuster model to do all adjustments."""
        self.__adjuster_model = adjuster_model

    def fit(
            self,
            views: Views,
            adjusted_view_def: AdjustedViewDef,
            sample_weight: Optional[SampleWeight] = None) -> ViewsAdjuster:
        res_adjusters = []
        for predictive_view_name in adjusted_view_def.predictive_view_names_seq():
            targeted_adjuster_model = TargetedViewAdjusterModel(
                adjusted_view=predictive_view_name,
                adjuster_views=adjusted_view_def.adjusters_for_view(view=predictive_view_name),
                view_adjuster_model=self.__adjuster_model)
            targeted_adjuster = targeted_adjuster_model.fit(views=views, sample_weight=sample_weight)
            res_adjusters.append(targeted_adjuster)
        return CompositeViewsAdjuster(adjusters=res_adjusters)

    def nick(self) -> str:
        return self.__adjuster_model.nick()

    def name(self) -> str:
        return self.__adjuster_model.name()

    def __str__(self) -> str:
        return str(self.__adjuster_model)


DEFAULT_VIEWS_ADJUSTER_MODEL = UniformViewsAdjusterModel(adjuster_model=DEFAULT_VIEW_ADJUSTER_MODEL)
DEFAULT_VIEWS_ADJUSTER_MODEL_NICK = DEFAULT_VIEWS_ADJUSTER_MODEL.nick()