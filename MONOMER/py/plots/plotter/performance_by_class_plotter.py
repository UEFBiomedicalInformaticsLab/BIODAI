import os
from typing import Sequence

from matplotlib.axes import Axes

from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR
from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.performance_by_class import one_performance_by_class_plot_to_ax, \
    read_all_num_features_external, read_all_cms
from plots.saved_hof import SavedHoF
from prediction_stats.confusion_matrix import PerformanceMeasure


class PerformanceByClassPlotter(HofsPlotter):
    """Works with external validation results."""
    __vertical_lines: Sequence[float]

    def __init__(self, vertical_lines: Sequence[float] = ()):
        self.__vertical_lines = vertical_lines

    def plot(self, ax: Axes, saved_hofs: Sequence[SavedHoF], color=None):
        hof = saved_hofs[0]  # Assumes just one hof.
        hof_dir = hof.path()
        num_features = read_all_num_features_external(hof_dir=hof_dir)
        n_solutions = len(num_features)
        if n_solutions > 0:
            cms = hof.confusion_matrices()
            one_performance_by_class_plot_to_ax(
                ax=ax,
                cms=cms,
                num_features=num_features,
                performance=PerformanceMeasure.balanced_accuracy,
                performance_name="balanced accuracy", legend_loc='lower right',
                vertical_lines=self.__vertical_lines)
