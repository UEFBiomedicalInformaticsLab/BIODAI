from typing import Sequence

from plots.hofs_plotter.hofs_plotter import HofsPlotterFrom2Cols
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.monotonic_front import multi_front_plot_ax
from plots.saved_hof import SavedHoF


class HofsBestTradePlotter(HofsPlotterFrom2Cols):

    def __init__(self, col_x: int, col_y: int, setup: PlotSetup):
        HofsPlotterFrom2Cols.__init__(self, col_x=col_x, col_y=col_y, setup=setup)

    def plot(self, ax, saved_hofs: Sequence[SavedHoF]):
        dfs = []
        names = []
        for s in saved_hofs:
            hof_dfs = s.to_dfs()
            if hof_dfs is not None:
                dfs.append(hof_dfs)
                names.append(s.name())
        setup = self._setup()
        if len(dfs) > 0:
            col_names = dfs[0][0].columns
            multi_front_plot_ax(
                ax=ax, dfs=dfs,
                col_x=self._col_x(), col_y=self._col_y(),
                col_name_x=col_names[self._col_x()], col_name_y=col_names[self._col_y()],
                names=names,
                x_min=setup.x_min(), y_min=setup.y_min(),
                x_max=setup.x_max(), y_max=setup.y_max())
