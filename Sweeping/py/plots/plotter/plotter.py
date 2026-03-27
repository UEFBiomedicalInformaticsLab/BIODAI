from abc import ABC, abstractmethod
from collections.abc import Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from plots.hofs_plotter.plot_setup import PlotSetup, DEFAULT_PLOT_SETUP
from plots.plot_utils import smart_save_fig
from util.printer.printer import Printer, OutPrinter


class Plotter(ABC):

    @abstractmethod
    def plot(self, ax: Axes, color=None):
        """Plots to ax. It is possible to suggest a color, it can be ignored depending on the specific plotter."""
        raise NotImplementedError()

    def name_parts(self) -> Sequence[str]:
        """Name parts for this plot."""
        return []


class PlotterWithSetup(Plotter, ABC):
    __setup: PlotSetup

    def __init__(self, setup: PlotSetup = DEFAULT_PLOT_SETUP):
        self.__setup = setup

    def _setup(self) -> PlotSetup:
        return self.__setup


def plotter_to_picture(plotter: Plotter, file: str, printer: Printer = OutPrinter()):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    plotter.plot(ax=ax)
    smart_save_fig(path=file, printer=printer)
