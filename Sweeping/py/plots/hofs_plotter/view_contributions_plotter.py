from collections.abc import Sequence

from matplotlib.axes import Axes

from plots.hofs_plotter.hofs_plotter import HofsPlotterWithSetup
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.saved_hof import SavedHoF
from plots.view_contributions import view_contributions_one_objective_to_ax


class ViewContributionsPlotter(HofsPlotterWithSetup):
    __objective_pos: int

    def __init__(self, objective_pos: int, setup: PlotSetup = PlotSetup()):
        HofsPlotterWithSetup.__init__(self=self, setup=setup)
        self.__objective_pos = objective_pos

    def plot(self, ax: Axes, saved_hofs: Sequence[SavedHoF], color=None):
        if len(saved_hofs) != 1:
            raise ValueError("This visualization is for one HoF alone.")
        hof = saved_hofs[0]
        objective_pos = self.__objective_pos
        objective_name =  hof.obj_nicks()[objective_pos]
        view_contributions_one_objective_to_ax(
            ax=ax, saved_solutions=hof.final_solutions(),
            objective_pos=objective_pos, objective_name=objective_name,
            view_names=hof.views_from_saved_setup(), setup=self._setup())
