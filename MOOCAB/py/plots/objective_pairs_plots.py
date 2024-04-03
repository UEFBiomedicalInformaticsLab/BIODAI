import os
from collections.abc import Sequence
from typing import Optional

from matplotlib import pyplot as plt
from pandas import DataFrame

from consts import FONT_SIZE
from plots.default_labels_map import LabelsTransformer, DUMMY_LABELS_TRANSFORMER
from plots.monotonic_front import df_vals_to_labels
from plots.plot_utils import smart_save_fig
from plots.saved_hof import SavedHoF
from saved_solutions.solution_attributes_archive import FITNESS
from util.dataframes import n_col
from util.plot_results import multiclass_scatter_to_ax, ADD_ELLIPSES_DEFAULT
from util.utils import names_by_differences


def external_objective_pairs_plot(ax, dfs: Sequence[DataFrame], i: int, j: int, label_i: str, label_j: str,
                                  names=Optional[Sequence[str]],
                                  x_min: float = None, x_max: float = None,
                                  y_min: float = None, y_max: float = None,
                                  alpha: float = None, font_size: int = FONT_SIZE,
                                  interpolate: bool = True, add_ellipses: bool = ADD_ELLIPSES_DEFAULT):
    """dfs is a sequence of dataframes from which to extract the x and y values.
    The x values are extracted from column i
    and the y values from column j"""
    x = []
    y = []
    for alg_dfs in dfs:
        alg_dfs = df_vals_to_labels(alg_dfs)
        x.append(alg_dfs.iloc[:, i])
        y.append(alg_dfs.iloc[:, j])
    multiclass_scatter_to_ax(
        ax=ax, x=x, y=y,
        x_label=label_i, y_label=label_j, class_labels=names,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, legend_loc="lower right", alpha=alpha,
        font_size=font_size, interpolate=interpolate, add_ellipses=add_ellipses)


def save_external_objective_pairs_plot(dfs: [DataFrame], i: int, j: int, label_i: str, label_j: str, save_path: str,
                                       names=Optional[Sequence[str]],
                                       x_min: float = None, x_max: float = None,
                                       y_min: float = None, y_max: float = None):
    fig_save_path = save_path + "/" + label_i + "_" + label_j + ".png"
    fig, ax = plt.subplots()
    external_objective_pairs_plot(ax=ax, dfs=dfs, i=i, j=j, label_i=label_i, label_j=label_j,
                                  names=names, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    smart_save_fig(path=fig_save_path)


def external_objective_pairs_plot_from_saved_hofs(
        saved_hofs: Sequence[SavedHoF], save_path: str,
        x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
        labels_map: LabelsTransformer = DUMMY_LABELS_TRANSFORMER):
    algo_dfs = []
    used_names = []
    for hof in saved_hofs:
        df = hof.to_df()
        if df is not None:
            algo_dfs.append(df)
            used_names.append(hof.name())
    if len(algo_dfs) > 0:
        n_objectives = n_col(algo_dfs[0])
        col_names = algo_dfs[0].columns
        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    label_i = col_names[i]
                    label_j = col_names[j]
                    label_i = labels_map.apply(label=label_i)
                    label_j = labels_map.apply(label=label_j)
                    save_external_objective_pairs_plot(
                        dfs=algo_dfs, i=i, j=j, label_i=label_i, label_j=label_j,
                        save_path=save_path, names=used_names,
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def one_objective_pair_plot_from_saved_hofs(
        ax, saved_hofs: Sequence[SavedHoF],
        i: int, j: int,
        x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
        labels_map: LabelsTransformer = DUMMY_LABELS_TRANSFORMER, alpha: float = None, font_size: int = FONT_SIZE,
        verbose: bool = False):
    algo_dfs = []
    name_parts = []
    for hof in saved_hofs:
        f = hof.path()
        if os.path.isdir(f):
            if hof.is_external():
                df = FITNESS.external_df(hof_dir=f)
            else:
                df = FITNESS.test_df(hof_dir=f)
            if df is not None:
                algo_dfs.append(df)
                name_parts.append(hof.name_parts())
            else:
                print("Unable to create dataframe from directory " + str(f))
        else:
            if verbose:
                print("path is not a directory: " + str(f))
    if len(algo_dfs) > 0:
        col_names = algo_dfs[0].columns
        label_i = col_names[i]
        label_j = col_names[j]
        label_i = labels_map.apply(label=label_i)
        label_j = labels_map.apply(label=label_j)
        external_objective_pairs_plot(
            ax=ax,
            dfs=algo_dfs, i=i, j=j, label_i=label_i, label_j=label_j,
            names=names_by_differences(object_features=name_parts),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, alpha=alpha, font_size=font_size)
