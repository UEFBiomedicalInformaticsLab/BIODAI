from typing import Optional

from model.sv_model import SampleWeight
from util.named import NickNamed
from util.table.table import Table
from view_adjuster.view_adjuster import ViewAdjuster
from view_adjuster.view_adjuster_model import ViewAdjusterModel
from views.adjusted_view_definition import print_view_and_adjusters
from views.views import Views


class TargetedViewAdjuster:
    __adjusted_view: str
    __adjuster_views: set[str]
    """Alphabetical order."""
    __view_adjuster: ViewAdjuster

    def __init__(self, adjusted_view: str, adjuster_views: set[str], view_adjuster: ViewAdjuster):
        self.__adjusted_view = adjusted_view
        self.__adjuster_views = adjuster_views
        self.__view_adjuster = view_adjuster

    def apply(self, views: Views) -> Table:
        """Applies the adjustment to the target view. Views must contain the adjuster views and also the view
        to be adjusted. The presence of additional views is not an issue."""
        return self.__view_adjuster.adjust(
            view_to_adjust=views.view(key=self.__adjusted_view),
            adjusting_views=views.select_views(view_names=self.__adjuster_views))

    def adjusted_view_name(self) -> str:
        return self.__adjusted_view


class TargetedViewAdjusterModel(NickNamed):
    __adjusted_view: str
    __adjuster_views: set[str]
    __view_adjuster_model: ViewAdjusterModel

    def __init__(self, adjusted_view: str, adjuster_views: set[str], view_adjuster_model: ViewAdjusterModel):
        self.__adjusted_view = adjusted_view
        self.__adjuster_views = adjuster_views
        self.__view_adjuster_model = view_adjuster_model

    def fit(self, views: Views, sample_weight: Optional[SampleWeight] = None) -> TargetedViewAdjuster:
        """Fits the adjustment to the target view. Views must contain the adjuster views and also the view
        to be adjusted. The presence of additional views is not an issue."""
        view_adjuster = self.__view_adjuster_model.fit(
            view_to_adjust=views.view(key=self.__adjusted_view),
            adjusting_views=views.select_views(view_names=self.__adjuster_views),
            sample_weight=sample_weight)
        return TargetedViewAdjuster(
            view_adjuster=view_adjuster,
            adjuster_views=self.__adjuster_views,
            adjusted_view=self.__adjusted_view)

    def nick(self) -> str:
        return self.__view_adjuster_model.nick() + "_" + print_view_and_adjusters(
            view_name=self.__adjusted_view, adjusters=self.__adjuster_views, compact=True)

    def name(self) -> str:
        return self.__view_adjuster_model.name() + "_" + print_view_and_adjusters(
            view_name=self.__adjusted_view, adjusters=self.__adjuster_views, compact=True)

    def __str__(self) -> str:
        return str(self.__view_adjuster_model) + " applied to " + print_view_and_adjusters(
            view_name=self.__adjusted_view, adjusters=self.__adjuster_views, compact=False)
