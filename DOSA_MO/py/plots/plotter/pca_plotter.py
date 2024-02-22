from matplotlib.axes import Axes
from pandas import DataFrame

from plots.plot_utils import pca2d_view_ax
from plots.plotter.plotter import Plotter


class PcaPlotter(Plotter):
    __view_data: DataFrame
    __outcome_data: DataFrame

    def __init__(self, view_data: DataFrame, outcome_data: DataFrame):
        self.__view_data = view_data
        self.__outcome_data = outcome_data

    def plot(self, ax: Axes, color=None):
        pca2d_view_ax(ax=ax, view=self.__view_data, outcome=self.__outcome_data,
                      show_counts=True, order_by_counts=False, point_size=10, legend_loc='upper right')
