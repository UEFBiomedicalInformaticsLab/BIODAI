import collections
import os
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from consts import FONT_SIZE, FINAL_STR
from cross_validation.multi_objective.cross_evaluator.hof_saver import SOLUTION_FEATURES_PREFIX, \
    SOLUTION_FEATURES_EXTENSION
from external_validation.mo_external_evaluator.hof_saver import SOLUTION_FEATURES_EXTERNAL
from input_data.input_data import InputData
from plots.saved_hof import is_external_dir
from util.printer.printer import Printer, OutPrinter


COLORS = ['b', 'r', 'g', 'y', 'm', 'c', 'tab:orange', 'tab:brown', 'tab:gray', 'k']


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
        x_max: float = None, y_max: float = None):
    with plt.style.context({'font.size': FONT_SIZE}):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        lines = []
        n_lines = len(y)
        if line_labels is None:
            line_labels = [str(i) for i in range(n_lines)]
        for i in range(n_lines):
            color = COLORS[i % len(COLORS)]
            line = ax.plot(x[i], y[i], "-", color=str(color), label=str(line_labels[i]))
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
                    x_max: float = None, y_max: float = None):
    fig, ax = plt.subplots()
    multi_line_plot_ax(ax=ax, y=y, x=x, line_labels=line_labels, x_label=x_label, y_label=y_label,
                       x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


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


def pca2d_view(view: DataFrame, view_name: str,
               outcome: DataFrame, outcome_name: str,
               directory: str, printer: Printer = OutPrinter(),
               show_counts: bool = True):
    if len(outcome.columns) == 1:
        counter = collections.Counter(outcome.iloc[:, 0]).most_common()
        targets = [c[0] for c in counter]
        if len(targets) <= len(COLORS):
            pc1_str = 'principal component 1'
            pc2_str = 'principal component 2'
            view = copy(view)
            view = view.dropna(axis=1)  # PCA does not work with NA
            view = StandardScaler().fit_transform(view)
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(view)
            principal_df = pd.DataFrame(
                data=principal_components, columns=[pc1_str, pc2_str])
            final_df = pd.concat([principal_df, outcome], axis=1)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=FONT_SIZE)
            ax.set_ylabel('Principal Component 2', fontsize=FONT_SIZE)
            for target, color in zip(targets, COLORS):
                indices_to_keep = (outcome.iloc[:, 0] == target)
                x = final_df.loc[indices_to_keep, pc1_str]
                y = final_df.loc[indices_to_keep, pc2_str]
                ax.scatter(x, y,
                           c=color,
                           s=50)
            ax.legend(targets)
            if show_counts:
                target_labels = [str(c[0]) + " (" + str(c[1]) + ")" for c in counter]
                ax.legend(labels=target_labels)
            ax.grid()
            smart_save_fig(
                path=os.path.join(directory, view_name + "_" + outcome_name + "_pca" + ".png"), printer=printer)
        else:
            printer.print("PCA not plotted for view " + str(view_name) + " and outcome " + str(outcome_name) +
                          ": too many outcome classes.")
    else:
        printer.print("PCA not plotted for view " + str(view_name) + " and outcome " + str(outcome_name) +
                      ": outcome has not exactly 1 column.")


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


def feature_counts_from_df(df: DataFrame) -> Sequence[int]:
    return [sum(row) for row in df.values.tolist()]


def hof_used_features_fold_dfs(hof_dir: str) -> Sequence[DataFrame]:
    """Fold dfs are returned in alphabetic order"""
    files = []
    if is_external_dir(hof_dir=hof_dir):
        for file in os.listdir(hof_dir):
            if file == SOLUTION_FEATURES_EXTERNAL:
                files.append(file)
    else:
        for file in os.listdir(hof_dir):
            if file.startswith(SOLUTION_FEATURES_PREFIX) and file.endswith(SOLUTION_FEATURES_EXTENSION):
                if FINAL_STR not in file:
                    files.append(file)
    files.sort()
    return [pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, f)) for f in files]


def hof_folds_feature_counts(hof_dir: str) -> Sequence[Sequence[int]]:
    """Folds are returned in alphabetic order. For each fold returns a sequence of feature counts for the solutions."""
    dfs = hof_used_features_fold_dfs(hof_dir=hof_dir)
    res = []
    for df in dfs:
        res.append(feature_counts_from_df(df=df))
    return res
