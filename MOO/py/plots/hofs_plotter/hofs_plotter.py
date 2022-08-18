from abc import ABC, abstractmethod
from collections.abc import Sequence

from plots.hofs_plotter.plot_setup import PlotSetup
from plots.saved_hof import SavedHoF


class HofsPlotter(ABC):

    @abstractmethod
    def plot(self, ax, saved_hofs: Sequence[SavedHoF]):
        """Plots to ax."""
        raise NotImplementedError()


class HofsPlotterWithSetup(HofsPlotter, ABC):
    __setup: PlotSetup

    def __init__(self, setup: PlotSetup):
        self.__setup = setup

    def _setup(self) -> PlotSetup:
        return self.__setup


class HofsPlotterFrom2Cols(HofsPlotterWithSetup, ABC):
    __col_x: int
    __col_y: int

    def __init__(self, col_x: int, col_y: int, setup: PlotSetup):
        HofsPlotterWithSetup.__init__(self, setup=setup)
        self.__col_x = col_x
        self.__col_y = col_y

    def _col_x(self) -> int:
        return self.__col_x

    def _col_y(self) -> int:
        return self.__col_y
