from collections.abc import Sequence

from plots.hofs_plotter.hofs_plotter import HofsPlotterFrom2Cols
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.monotonic_front import sequence_vals_to_labels
from plots.saved_hof import SavedHoF
from util.plot_results import multiclass_scatter_to_ax
from util.sequence_utils import flatten_iterable_of_iterable


class HofsDetailedPlotter(HofsPlotterFrom2Cols):

    def __init__(self, col_x: int, col_y: int, setup: PlotSetup = PlotSetup()):
        HofsPlotterFrom2Cols.__init__(self, col_x=col_x, col_y=col_y, setup=setup)

    def plot(self, ax, saved_hofs: Sequence[SavedHoF], color=None):
        setup = self._setup()
        if len(saved_hofs) == 0:
            return
        if len(saved_hofs) > 1:
            raise ValueError("Only one hof is supported.")
        hof = saved_hofs[0]
        x_train_fitness = hof.train_fitness_objective_folds(obj=self._col_x())
        y_train_fitness = hof.train_fitness_objective_folds(obj=self._col_y())
        if x_train_fitness is None or y_train_fitness is None:
            raise ValueError("Missing expected fitness.")
        x_expected = flatten_iterable_of_iterable(x_train_fitness)
        y_expected = flatten_iterable_of_iterable(y_train_fitness)
        x_measured = hof.test_fitness_objective(obj=self._col_x())
        y_measured = hof.test_fitness_objective(obj=self._col_y())
        if x_measured is None or y_measured is None:
            raise ValueError("Missing measured fitness.")
        labels_map = setup.labels_map()
        obj_nicks = hof.obj_nicks()
        x_label = obj_nicks[self._col_x()]
        y_label = obj_nicks[self._col_y()]
        x_expected = sequence_vals_to_labels(s=x_expected, label=x_label)
        y_expected = sequence_vals_to_labels(s=y_expected, label=y_label)
        x_measured = sequence_vals_to_labels(s=x_measured, label=x_label)
        y_measured = sequence_vals_to_labels(s=y_measured, label=y_label)
        multiclass_scatter_to_ax(
            ax=ax,
            x=[x_expected, x_measured],
            y=[y_expected, y_measured],
            x_label = labels_map.apply(x_label), y_label = labels_map.apply(y_label),
            class_labels = labels_map.apply_all(["expected", "measured"]),
            colors = setup.palette().colors(n_colors=2),
            x_min=setup.x_min(), x_max=setup.x_max(), y_min=setup.y_min(), y_max=setup.y_max(),
            alpha=setup.alpha(), font_size=setup.font_size(),
            decimals=None,  # Setting decimals might not look good in this case.
            interpolate=setup.splines(),
            legend_loc="lower right")
        # ax.set_title(hof.name())
