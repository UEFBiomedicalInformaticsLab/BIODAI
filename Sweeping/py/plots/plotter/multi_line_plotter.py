from collections.abc import Sequence
from typing import Optional

from matplotlib.axes import Axes

from consts import FONT_SIZE
from plots.plot_utils import DEFAULT_DECIMALS, multi_line_plot_ax
from plots.plotter.plotter import Plotter


class MultiLinePlotter(Plotter):
    __y: Sequence[Sequence[float]]
    __x: Optional[Sequence[Sequence[float]]]
    __line_labels: Optional[Sequence[str]]
    __x_label: str
    __y_label: str
    __x_min: Optional[float]
    __y_min: Optional[float]
    __x_max: Optional[float]
    __y_max: Optional[float]
    __font_size: int
    __colors_by_label: bool
    __decimals: Optional[int]
    __x_tick_labels: Optional[Sequence[str]]
    __legend_loc: str

    def __init__(
            self,
            y: Sequence[Sequence[float]],
            x: Optional[Sequence[Sequence[float]]],
            line_labels: Optional[Sequence[str]] = None,
            x_label: str = "x",
            y_label: str = "y",
            x_min: Optional[float] = None, y_min: Optional[float] = None,
            x_max: Optional[float] = None, y_max: Optional[float] = None,
            font_size: int = FONT_SIZE,
            colors_by_label: bool = True,
            decimals: Optional[int] = DEFAULT_DECIMALS,
            x_tick_labels: Optional[Sequence[str]] = None,
            legend_loc: str = "best"):
        self.__y = y
        self.__x = x
        self.__line_labels = line_labels
        self.__x_label = x_label
        self.__y_label = y_label
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__font_size = font_size
        self.__colors_by_label = colors_by_label
        self.__decimals = decimals
        self.__x_tick_labels = x_tick_labels
        self.__legend_loc = legend_loc

    def plot(self, ax: Axes, color=None):
        multi_line_plot_ax(
            ax=ax, y=self.__y, x=self.__x,
            line_labels=self.__line_labels,
            x_label=self.__x_label, y_label=self.__y_label,
            x_min=self.__x_min, y_min=self.__y_min,
            x_max=self.__x_max, y_max=self.__y_max,
            font_size=self.__font_size, colors_by_label=self.__colors_by_label,
            decimals=self.__decimals,
            x_tick_labels=self.__x_tick_labels,
            legend_loc=self.__legend_loc)
