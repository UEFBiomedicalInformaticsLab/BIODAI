from typing import Sequence

from plots.hofs_plotter.hofs_plotter import HofsPlotterFrom2Cols
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.objective_pairs_plots import one_objective_pair_plot_from_saved_hofs
from plots.saved_hof import SavedHoF


class HofsScatterplotter(HofsPlotterFrom2Cols):

    def __init__(self, col_x: int, col_y: int, setup: PlotSetup):
        HofsPlotterFrom2Cols.__init__(self, col_x=col_x, col_y=col_y, setup=setup)

    def plot(self, ax, saved_hofs: Sequence[SavedHoF], color=None):
        setup = self._setup()
        one_objective_pair_plot_from_saved_hofs(
            ax=ax, saved_hofs=saved_hofs, i=self._col_x(), j=self._col_y(),
            x_min=setup.x_min(), x_max=setup.x_max(), y_min=setup.y_min(), y_max=setup.y_max(),
            labels_map=setup.labels_map(), alpha=setup.alpha(), font_size=setup.font_size())
