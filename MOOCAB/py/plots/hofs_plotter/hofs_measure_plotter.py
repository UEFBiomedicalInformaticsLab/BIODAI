import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from plots.barplot import barplot_with_std_ax, barplot_ax
from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.saved_hof import SavedHoF
from saved_solutions.run_measure.run_fold_measure import RunFoldMeasure
from saved_solutions.run_measure.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solutions_from_files import solutions_from_files


class HofsMeasurePlotter(HofsPlotter, ABC):

    def __init__(self):
        HofsPlotter.__init__(self)

    def plot(self, ax, saved_hofs: Sequence[SavedHoF], color=None):
        solutions = []
        hof_names = []
        hof_classes = []
        for alg_hofs in saved_hofs:
            h_path = alg_hofs.path()
            if os.path.isdir(h_path):
                solutions.append(solutions_from_files(hof_dir=h_path))
                hof_names.append(alg_hofs.name())
                hof_classes.append(alg_hofs.main_algorithm_label())
        self._inner_plot(ax, solutions, hof_names, self.measure_name(), color=color, hof_classes=hof_classes)

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

    def __init__(self, measure: RunFoldMeasure):
        HofsMeasurePlotter.__init__(self)
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
            classes=hof_classes)


class HofsRunMeasurePlotter(HofsMeasurePlotter):
    __measure: RunMeasure

    def __init__(self, measure: RunMeasure):
        HofsMeasurePlotter.__init__(self)
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
