from __future__ import annotations
from typing import Optional

from consts import FONT_SIZE
from plots.default_labels_map import LabelsTransformer, DUMMY_LABELS_TRANSFORMER, DEFAULT_LABELS_TRANSFORMER, \
    NO_LOG_LABELS_TRANSFORMER
from plots.hofs_plotter.palette import Palette, DEFAULT_PALETTE
from plots.plot_utils import DEFAULT_DECIMALS


DEFAULT_SPLINES = True


class PlotSetup:
    __x_min: Optional[float]
    __x_max: Optional[float]
    __y_min: Optional[float]
    __y_max: Optional[float]
    __alpha: Optional[float]
    __labels_map: LabelsTransformer
    __font_size: int
    __decimals: Optional[int]
    __splines: bool
    __palette: Palette
    __legend_loc: str

    def __init__(self,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 alpha: Optional[float] = None,
                 labels_map: LabelsTransformer = DUMMY_LABELS_TRANSFORMER,
                 font_size: int = FONT_SIZE,
                 decimals: Optional[int] = DEFAULT_DECIMALS,
                 splines: bool = DEFAULT_SPLINES,
                 palette: Palette = DEFAULT_PALETTE,
                 legend_loc: str = "best"):
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__alpha = alpha
        self.__labels_map = labels_map
        self.__font_size = font_size
        self.__decimals = decimals
        self.__splines = splines
        self.__palette = palette
        self.__legend_loc = legend_loc

    def x_min(self) -> Optional[float]:
        return self.__x_min

    def x_max(self) -> Optional[float]:
        return self.__x_max

    def y_min(self) -> Optional[float]:
        return self.__y_min

    def y_max(self) -> Optional[float]:
        return self.__y_max

    def alpha(self) -> Optional[float]:
        return self.__alpha

    def labels_map(self) -> LabelsTransformer:
        return self.__labels_map

    def font_size(self) -> int:
        return self.__font_size

    def label_transform(self, label: str) -> str:
        return self.__labels_map.apply(label=label)

    def decimals(self) -> Optional[int]:
        return self.__decimals

    def set_decimals(self, decimals: Optional[int]) -> PlotSetup:
        return PlotSetup(
            x_min=self.x_min(), x_max=self.x_max(),
            y_min=self.y_min(), y_max=self.y_max(),
            alpha=self.alpha(),
            labels_map=self.labels_map(),
            font_size=self.font_size(),
            decimals=decimals,
            splines=self.splines(),
            palette=self.palette())

    def splines(self) -> bool:
        return self.__splines

    def palette(self) -> Palette:
        return self.__palette

    def set_palette(self, palette: Palette) -> PlotSetup:
        return PlotSetup(
            x_min=self.x_min(), x_max=self.x_max(),
            y_min=self.y_min(), y_max=self.y_max(),
            alpha=self.alpha(),
            labels_map=self.labels_map(),
            font_size=self.font_size(), decimals=self.decimals(),
            splines=self.splines(),
            palette=palette)

    def set_labels_map(self, labels_map: LabelsTransformer) -> PlotSetup:
        return PlotSetup(
            x_min=self.x_min(), x_max=self.x_max(),
            y_min=self.y_min(), y_max=self.y_max(),
            alpha=self.alpha(),
            labels_map=labels_map,
            font_size=self.font_size(), decimals=self.decimals(),
            splines=self.splines(),
            palette=self.palette())

    def legend_loc(self) -> str:
        return self.__legend_loc


class PlotSetupWithDefaultLabels(PlotSetup):

    def __init__(self,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None,
                 alpha: Optional[float] = None,
                 font_size: int = FONT_SIZE):
        PlotSetup.__init__(
            self=self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, alpha=alpha, font_size=font_size,
            labels_map=DEFAULT_LABELS_TRANSFORMER)


DEFAULT_PLOT_SETUP = PlotSetupWithDefaultLabels()
NO_LOG_PLOT_SETUP = PlotSetupWithDefaultLabels().set_labels_map(labels_map=NO_LOG_LABELS_TRANSFORMER)
