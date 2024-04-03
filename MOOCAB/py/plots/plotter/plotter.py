from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from plots.plot_utils import smart_save_fig
from util.printer.printer import Printer, OutPrinter


class Plotter(ABC):

    @abstractmethod
    def plot(self, ax: Axes, color=None):
        """Plots to ax."""
        raise NotImplementedError()


def plotter_to_picture(plotter: Plotter, file: str, printer: Printer = OutPrinter()):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    plotter.plot(ax=ax)
    smart_save_fig(path=file, printer=printer)
