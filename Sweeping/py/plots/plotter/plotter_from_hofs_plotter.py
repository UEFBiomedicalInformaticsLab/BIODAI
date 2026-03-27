from collections.abc import Sequence

from matplotlib.axes import Axes

from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.plotter.plotter import Plotter
from plots.saved_hof import SavedHoF, battery_common_name_parts


class PlotterFromHofsPlotter(Plotter):
    __hofs_plotter: HofsPlotter
    __saved_hofs: Sequence[SavedHoF]

    def __init__(self, hofs_plotter: HofsPlotter, saved_hofs: Sequence[SavedHoF]):
        self.__hofs_plotter = hofs_plotter
        self.__saved_hofs = saved_hofs

    def plot(self, ax: Axes, color=None):
        self.__hofs_plotter.plot(ax=ax, saved_hofs=self.__saved_hofs, color=color)

    def name_parts(self) -> Sequence[str]:
        return battery_common_name_parts(hofs=self.__saved_hofs)
