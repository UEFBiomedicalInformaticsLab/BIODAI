import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from plots.barplot import barplot_with_std_ax, barplot_ax
from plots.hofs_plotter.hofs_plotter import HofsPlotterWithSetup
from plots.hofs_plotter.plot_setup import PlotSetupWithDefaultLabels, PlotSetup
from plots.saved_hof import SavedHoF
from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.run_measure.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solutions_from_files import solutions_from_files
from util.name_parts import names_by_differences


class HofsMeasurePlotter(HofsPlotterWithSetup, ABC):
    """Only label transform is used from the PlotSetup."""

    def __init__(self, setup: PlotSetup = PlotSetupWithDefaultLabels()):
        HofsPlotterWithSetup.__init__(self, setup=setup)

    def plot(self, ax, saved_hofs: Sequence[SavedHoF], color=None):
        solutions = []
        hof_classes = []
        name_parts = []
        for alg_hofs in saved_hofs:
            h_path = alg_hofs.path()
            if os.path.isdir(h_path):
                solutions.append(solutions_from_files(hof_dir=h_path))
                name_parts.append(alg_hofs.name_parts())
                hof_classes.append(alg_hofs.main_algorithm_label())
        labels_map = self._setup().labels_map()
        hof_names = labels_map.apply_all(names_by_differences(object_features=name_parts))
        measure_name = labels_map.apply(label=self.measure_name())
        self._inner_plot(ax, solutions, hof_names, measure_name, color=color, hof_classes=hof_classes)

    @abstractmethod
    def _inner_plot(self, ax, solutions, hof_names, measure_name, color=None, hof_classes: Optional[Sequence] = None):
        raise NotImplementedError()

    @abstractmethod
    def measure_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def measure_nick(self) -> str:
        raise NotImplementedError()


class HofsFoldMeasurePlotter(HofsMeasurePlotter):
    __measure: RunFoldMeasure

    def __init__(self, measure: RunFoldMeasure, setup: PlotSetup = PlotSetupWithDefaultLabels()):
        HofsMeasurePlotter.__init__(self, setup=setup)
        self.__measure = measure

    def measure_name(self) -> str:
        return self.__measure.name()

    def measure_nick(self) -> str:
        return self.__measure.nick()

    def _inner_plot(self, ax, solutions: list[Sequence[Sequence[SavedSolution]]], hof_names, measure_name, color=None,
                    hof_classes: Optional[Sequence] = None):
        barplot_with_std_ax(
            ax=ax,
            measures=[self.__measure.compute_measures(solutions=s) for s in solutions],
            bar_names=hof_names,
            value_label=self.measure_name(),
            bar_color=color,
            classes=hof_classes,
            labels_transformer=self._setup().labels_map())


class HofsRunMeasurePlotter(HofsMeasurePlotter):
    __measure: RunMeasure

    def __init__(self, measure: RunMeasure, setup: PlotSetup = PlotSetupWithDefaultLabels()):
        HofsMeasurePlotter.__init__(self, setup=setup)
        self.__measure = measure

    def measure_name(self) -> str:
        return self.__measure.name()

    def measure_nick(self) -> str:
        return self.__measure.nick()

    def _inner_plot(self, ax, solutions: list[Sequence[Sequence[SavedSolution]]], hof_names, measure_name, color=None,
                    hof_classes: Optional[Sequence] = None):
        barplot_ax(
            ax=ax,
            bar_lengths=[self.__measure.compute_measure(solutions=s) for s in solutions],
            bar_names=hof_names,
            label_y=self.measure_name(),
            bar_color=color,
            classes=hof_classes
        )
