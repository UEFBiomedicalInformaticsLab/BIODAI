import os
from copy import copy

from matplotlib.axes import Axes
from pandas import DataFrame

from input_data.input_data import InputData
from input_data.outcome_type import OutcomeType
from util.survival.survival_utils import survival_times, survival_events, SURVIVAL_DURATION_STR
from plots.counts_over_steps import weights_over_steps_plot_ax
from plots.plotter.plotter import Plotter, plotter_to_picture
from util.dataframes import n_row
from util.printer.printer import Printer, OutPrinter
from util.sequence_utils import stable_uniques, transpose


def survival_plot_ax(ax: Axes, outcome_data: DataFrame):
    labels = ("deceased", "censored", "surviving")
    data = outcome_data.sort_values(by=[SURVIVAL_DURATION_STR], inplace=False)
    n_individuals = n_row(data)
    times = survival_times(data)
    events = survival_events(data)
    x = stable_uniques([0.0] + data[SURVIVAL_DURATION_STR])
    num_steps = len(x)
    counts = [[0, 0, n_individuals]]
    row = 0
    for i in range(num_steps):
        x_i = x[i]
        if i > 0:
            counts.append(copy(counts[i-1]))
        while row < n_individuals and times[row] == x_i:
            counts[i][2] -= 1
            if events[row]:
                counts[i][0] += 1
            else:
                counts[i][1] += 1
            row += 1
    counts = transpose(counts)
    weights_over_steps_plot_ax(ax=ax, counts=counts, labels=labels, x=x, x_label="time", y_label="individuals")


class SurvivalPlotter(Plotter):
    __outcome_data: DataFrame

    def __init__(self, outcome_data: DataFrame):
        self.__outcome_data = outcome_data

    def plot(self, ax: Axes, color=None):
        survival_plot_ax(ax=ax, outcome_data=self.__outcome_data)


def plot_all_survival_outcomes(input_data: InputData, directory: str, printer: Printer = OutPrinter()):
    for o in input_data.outcomes():
        if o.type() == OutcomeType.survival:
            save_file = os.path.join(directory, o.name() + "_stackplot" + ".png")
            printer.print("Plotting survival outcome " + o.name() + " to " + save_file)
            plotter = SurvivalPlotter(outcome_data=o.data())
            plotter_to_picture(plotter=plotter, file=save_file, printer=printer)
