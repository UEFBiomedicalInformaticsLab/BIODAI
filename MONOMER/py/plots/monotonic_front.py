import os
from collections.abc import Sequence
from copy import copy
from typing import Optional

from matplotlib import pyplot as plt
from pandas import DataFrame

from consts import FONT_SIZE
from objective.objective_computer import Leanness
from plots.plot_utils import single_line_plot_to_path, multi_line_plot_ax, smart_save_fig
from plots.saved_hof import SavedHoF
from plots.hof_utils import test_dfs
from util.dataframes import n_col
from util.summer import KahanSummer


def single_line_from_front_lines(
        x: Sequence[Sequence[float]], y: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    res_x = set()
    for xi in x:
        res_x = res_x.union(xi)
    res_x = list(res_x)
    res_x.sort()
    n_folds = len(x)
    y_folds = []
    for f in range(n_folds):
        x_f = x[f]
        y_f = y[f]
        x_f, y_f = (list(t) for t in zip(*sorted(zip(x_f, y_f))))  # Sorts by x with y to break ties.
        fold_y = [-1] * len(res_x)
        pos = 0
        for i in range(len(res_x)):
            res_x_i = res_x[i]
            while pos < len(x_f) and x_f[pos] < res_x_i:
                pos += 1
            if pos < len(x_f):
                fold_y[i] = y_f[pos]
        y_folds.append(fold_y)
    res_y = [-1] * len(res_x)
    for i in range(len(res_x)):
        y_vals = [yf[i] for yf in y_folds]
        if -1 not in y_vals:
            res_y[i] = KahanSummer.mean(y_vals)
    pruned_x = []
    pruned_y = []
    for i in range(len(res_x)):
        if res_y[i] != -1:
            pruned_x.append(res_x[i])
            pruned_y.append(res_y[i])
    return pruned_x, pruned_y


def remove_dominated(x: Sequence[float], y: Sequence[float]) -> (Sequence[float], Sequence[float]):
    res_x = []
    res_y = []
    for i in range(len(x)):
        y_i = y[i]
        k = True
        for j in range(i + 1, len(x)):
            if y_i <= y[j]:
                k = False
        if k:
            res_x.append(x[i])
            res_y.append(y[i])
    return res_x, res_y


def single_line_from_sequences(x: [[float]], y: [[float]], name_x: str, name_y: str) -> ([float], [float]):
    single_x, single_y = single_line_from_front_lines(x=x, y=y)
    single_x, single_y = remove_dominated(x=single_x, y=single_y)
    single_x = sequence_vals_to_labels(s=single_x, label=name_x)
    single_y = sequence_vals_to_labels(s=single_y, label=name_y)
    return single_x, single_y


def single_front_plot_from_sequences(x: [[float]], y: [[float]], label_x: str, label_y: str, hof_dir: str):
    single_x, single_y = single_line_from_sequences(x=x, y=y, name_x=label_x, name_y=label_y)
    single_line_plot_to_path(y=single_y, x=single_x, y_label=label_y, x_label=label_x,
                             plot_path=hof_dir + "/" + label_x + "_" + label_y + ".png")


def multi_front_plot_ax(
        ax, dfs: Sequence[Sequence[DataFrame]], col_x: int, col_y: int, col_name_x: str, col_name_y: str,
        names=Optional[Sequence[str]], x_label: str = None, y_label: str = None,
        x_min: float = None, y_min: float = None,
        x_max: float = None, y_max: float = None, font_size: int = FONT_SIZE):
    lines_x = []
    lines_y = []
    for alg_dfs in dfs:
        alg_x, alg_y = single_line_from_sequences(
            x=[df.iloc[:, col_x] for df in alg_dfs], y=[df.iloc[:, col_y] for df in alg_dfs],
            name_x=col_name_x, name_y=col_name_y)
        lines_x.append(alg_x)
        lines_y.append(alg_y)
    if x_label is None:
        x_label = col_name_x
    if y_label is None:
        y_label = col_name_y
    multi_line_plot_ax(
        ax=ax,
        x=lines_x, y=lines_y, x_label=x_label, y_label=y_label,
        line_labels=names, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, font_size=font_size)


def multi_front_plot(dfs: [[DataFrame]], i: int, j: int, col_name_x: str, col_name_y: str, save_path: str,
                     names=Optional[Sequence[str]], x_label: str = None, y_label: str = None,
                     x_min: float = None, y_min: float = None,
                     x_max: float = None, y_max: float = None):
    fig, ax = plt.subplots()
    multi_front_plot_ax(
        ax=ax, dfs=dfs, col_x=i, col_y=j, col_name_x=col_name_x, col_name_y=col_name_y,
        names=names, x_label=x_label, y_label=y_label,
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max)
    smart_save_fig(path=save_path + "/" + col_name_x + "_" + col_name_y + ".png")


def single_front_plot(dfs: [DataFrame], i: int, j: int, label_i: str, label_j: str, hof_dir: str):
    x = [df.iloc[:, i] for df in dfs]
    y = [df.iloc[:, j] for df in dfs]
    single_front_plot_from_sequences(x=x, y=y, label_x=label_i, label_y=label_j, hof_dir=hof_dir)


def sequence_vals_to_labels(s: Sequence[float], label: str) -> Sequence[float]:
    leanness = Leanness()
    if label.endswith(leanness.nick()):
        return leanness.vals_to_labels(s)
    else:
        return s


def df_vals_to_labels(df: DataFrame) -> DataFrame:
    col_names = df.columns
    res_df = copy(df)
    for c in col_names:
        res_df.loc[:, c] = sequence_vals_to_labels(res_df.loc[:, c], label=c)
    return res_df


def single_front_plots(hof_dir: str):
    dfs = test_dfs(hof_dir=hof_dir)
    n_objectives = n_col(dfs[0])
    col_names = dfs[0].columns
    if dfs is not None:
        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    single_front_plot(
                        dfs=dfs, i=i, j=j, label_i=col_names[i], label_j=col_names[j], hof_dir=hof_dir)


def single_front_plot_every_hof(main_hofs_dir: str):
    if os.path.isdir(main_hofs_dir):
        subfolders = [f.path for f in os.scandir(main_hofs_dir) if f.is_dir()]
        for f in subfolders:
            single_front_plots(hof_dir=f)


def multi_front_plot_from_dirs(hofs_dir: Sequence[str], save_path: str, names=Optional[Sequence[str]],
                               x_label: str = None, y_label: str = None,
                               x_min: float = None, y_min: float = None,
                               x_max: float = None, y_max: float = None
                               ):
    algo_dfs = []
    used_names = []
    for i in range(len(hofs_dir)):
        f = hofs_dir[i]
        name = str(i)
        if names is not None:
            name = names[i]
        if os.path.isdir(f):
            tdfs = test_dfs(hof_dir=f)
            if tdfs is not None:
                algo_dfs.append(tdfs)
                used_names.append(name)
            else:
                print("Unable to create dataframe from directory " + str(f))
        else:
            print("path is not a directory: " + str(f))
    if len(algo_dfs) > 0:
        n_objectives = n_col(algo_dfs[0][0])
        col_names = algo_dfs[0][0].columns
        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    label_i = col_names[i]
                    label_j = col_names[j]
                    multi_front_plot(
                        dfs=algo_dfs, i=i, j=j, col_name_x=label_i, col_name_y=label_j,
                        save_path=save_path, names=used_names, x_label=x_label, y_label=y_label,
                        x_min=x_min, y_min=y_min,
                        x_max=x_max, y_max=y_max)


def multi_front_plot_from_saved_hofs(saved_hofs: Sequence[SavedHoF], save_path: str,
                                     x_label: str = None, y_label: str = None,
                                     x_min: float = None, y_min: float = None,
                                     x_max: float = None, y_max: float = None):
    hofs_dir = [s.path() for s in saved_hofs]
    names = [s.name() for s in saved_hofs]
    multi_front_plot_from_dirs(hofs_dir=hofs_dir, save_path=save_path, names=names, x_label=x_label, y_label=y_label,
                               x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
