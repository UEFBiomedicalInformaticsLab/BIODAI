import collections
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from consts import FONT_SIZE
from input_data.input_data import InputData
from plots.plot_consts import COLORS, DEFAULT_PALETTE, PRINCIPAL_COMPONENT1_STR, PRINCIPAL_COMPONENT2_STR
from util.dataframes import n_row
from util.printer.printer import Printer, OutPrinter


def default_color_list(n_colors: int, desat: float = None, invert: bool = False):
    colors = seaborn.color_palette(palette=DEFAULT_PALETTE, n_colors=n_colors, desat=desat)
    if len(colors) < n_colors:
        colors = seaborn.color_palette(n_colors=n_colors, desat=desat)
    if invert:
        colors = colors[::-1]
    return colors


def smart_save_fig(path: str, printer: Printer = OutPrinter()):
    """If no extension is specified, .png is added automatically."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(path, bbox_inches='tight', dpi=600)
    except BaseException as e:
        printer.print("Exception while saving figure:\n" + str(e))
    plt.close()


def multi_line_plot_ax(
        ax, y: [[float]], x: [[float]], line_labels: [str] = None, x_label: str = "x", y_label: str = "y",
        x_min: float = None, y_min: float = None,
        x_max: float = None, y_max: float = None, font_size: int = FONT_SIZE):
    with plt.style.context({'font.size': font_size}):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        lines = []
        n_lines = len(y)
        if line_labels is None:
            line_labels = [str(i) for i in range(n_lines)]
        colors = default_color_list(n_colors=n_lines, invert=True)
        for i in range(n_lines):
            color = colors[i % len(colors)]
            line = ax.plot(x[i], y[i], "-", color=color, label=str(line_labels[i]))
            lines = lines + line
        if x_min is not None:
            plt.xlim(left=x_min)
        if x_max is not None:
            plt.xlim(right=x_max)
        if y_min is not None:
            plt.ylim(bottom=y_min)
        if y_max is not None:
            plt.ylim(top=y_max)
        ax.grid(b=True)
        labs = [line.get_label() for line in lines]
        ax.legend(lines, labs, loc="lower right")


def multi_line_plot(y: [[float]], x: [[float]], line_labels: [str] = None, x_label: str = "x", y_label: str = "y",
                    x_min: float = None, y_min: float = None,
                    x_max: float = None, y_max: float = None, font_size: int = FONT_SIZE):
    fig, ax = plt.subplots()
    multi_line_plot_ax(ax=ax, y=y, x=x, line_labels=line_labels, x_label=x_label, y_label=y_label,
                       x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, font_size=font_size)


def multi_line_plot_to_path(
        y: [[float]], x: [[float]], path: str, line_labels: [str] = None, x_label: str = "x", y_label: str = "y",
        x_min: float = None, y_min: float = None,
        x_max: float = None, y_max: float = None):
    multi_line_plot(x=x, y=y, line_labels=line_labels, x_label=x_label, y_label=y_label,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    smart_save_fig(path=path)


def single_line_plot(
        y: [float], x: Optional[Sequence[float]] = None, x_label: str = "x", y_label: str = "y"):
    if x is None:
        x = range(1, len(y)+1)
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    lines = []
    lines = lines + ax.plot(
        x, y, "b-", label=y_label)
    plt.grid()
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc="center right")


def single_line_plot_to_path(
        y: [float], plot_path: str, x: Optional[Sequence[float]] = None, x_label: str = "x", y_label: str = "y"):
    single_line_plot(y=y, x=x, x_label=x_label, y_label=y_label)
    smart_save_fig(path=plot_path)


def pca2d_view_ax(ax: Axes,
                  view: DataFrame,
                  outcome: DataFrame,
                  show_counts: bool = True,
                  font_size: int = FONT_SIZE,
                  order_by_counts: bool = True,
                  point_size: float = 50,
                  legend_loc: str = 'best'):
    if n_row(view) != n_row(outcome):
        raise ValueError(
            "View and outcome dataframes do not have the same number of rows.\n" +
            "View rows: " + str(n_row(view)) + "\n" +
            "Outcome rows: " + str(n_row(outcome)) + "\n")
    rows_to_keep = view.dropna().index
    view = view.iloc[rows_to_keep, :]  # PCA does not work with NA
    outcome = outcome.iloc[rows_to_keep, :]
    if len(outcome.columns) == 1:
        counter = collections.Counter(outcome.iloc[:, 0]).most_common()
        targets = [c[0] for c in counter]
        if not order_by_counts:
            targets.sort()
            counter.sort()
        n_targets = len(targets)
        colors = default_color_list(n_colors=n_targets, invert=False)
        n_colors = len(colors)
        if n_targets <= n_colors and n_targets <= 20:
            pc1_str = PRINCIPAL_COMPONENT1_STR
            pc2_str = PRINCIPAL_COMPONENT2_STR
            pca = PCA(n_components=2)
            view = StandardScaler().fit_transform(view)
            principal_components = pca.fit_transform(view)
            principal_df = pd.DataFrame(
                data=principal_components, columns=[pc1_str, pc2_str])
            final_df = pd.concat([principal_df, outcome], axis=1)
            with plt.style.context({'font.size': font_size}):
                ax.set_xlabel(pc1_str, fontsize=font_size)
                ax.set_ylabel(pc2_str, fontsize=font_size)
                for target, color in zip(targets, colors):
                    indices_to_keep = (outcome.iloc[:, 0] == target)
                    x = final_df.loc[indices_to_keep, pc1_str]
                    y = final_df.loc[indices_to_keep, pc2_str]
                    ax.scatter(x, y, c=color, s=point_size)
                if show_counts:
                    legend_entries = [str(c[0]) + " (" + str(c[1]) + ")" for c in counter]
                else:
                    legend_entries = targets
                ax.legend(legend_entries, loc=legend_loc)
                ax.grid()
        else:
            raise ValueError("Too many outcome classes.")
    else:
        raise ValueError("Outcome has not exactly 1 column.")


def pca2d_view(view: DataFrame, view_name: str,
               outcome: DataFrame, outcome_name: str,
               directory: str, printer: Printer = OutPrinter(),
               show_counts: bool = True,
               font_size: int = FONT_SIZE):
    with plt.style.context({'font.size': font_size}):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        pca2d_view_ax(ax=ax, view=view, outcome=outcome, show_counts=show_counts, font_size=font_size)
        try:
            smart_save_fig(
                path=os.path.join(directory, view_name + "_" + outcome_name + "_pca" + ".png"), printer=printer)
        except ValueError as e:
            printer.print("PCA not plotted for view " + str(view_name) + " and outcome " + str(outcome_name) + "\n" +
                          str(e))


def pca2d(data: InputData, directory: str, printer: Printer = OutPrinter()):
    for outcome in data.outcomes():
        outcome_data = outcome.data()
        if len(outcome_data.columns) == 1:
            if len(set(outcome_data.iloc[:, 0])) <= len(COLORS):
                outcome_name = outcome.name()
                for view_name in data.views():
                    view_data = data.view(view_name)
                    pca2d_view(view=view_data, view_name=view_name, outcome=outcome_data, outcome_name=outcome_name,
                               directory=directory, printer=printer)
