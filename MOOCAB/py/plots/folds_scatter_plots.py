import os
from collections.abc import Sequence
from typing import Optional

from matplotlib import pyplot as plt
from pandas import DataFrame

from plots.monotonic_front import df_vals_to_labels
from plots.plot_utils import smart_save_fig
from plots.saved_hof import SavedHoF
from saved_solutions.solution_attributes_archive import FITNESS
from util.dataframes import select_columns_by_prefix, n_col
from util.plot_results import multiclass_scatter_to_ax, ADD_ELLIPSES_DEFAULT


def get_column(df: DataFrame, prefix: str, suffix: str) -> Sequence:
    colname = prefix + "_" + suffix
    if colname in df.columns:
        return df[colname]
    else:
        raise ValueError("Column not found, column names: " + str(df.columns))


def algorithm_folds_scatter_plot(ax, fold_dfs: Sequence[DataFrame], x_objective: str, y_objective: str,
                                 x_min: Optional[float] = None, y_min: Optional[float] = None,
                                 x_max: Optional[float] = None, y_max: Optional[float] = None,
                                 alpha: Optional[float] = None,
                                 x_label: str = None, y_label: str = None,
                                 interpolate: bool = True, add_ellipses: bool = ADD_ELLIPSES_DEFAULT):
    has_inner_cv = n_col(select_columns_by_prefix(df=fold_dfs[0], prefix="inner_cv")) > 0
    x = [[], [], []]
    y = [[], [], []]
    class_labels = ["train", "inner cv", "test"]
    for df in fold_dfs:
        df = df_vals_to_labels(df=df)
        x[0].extend(get_column(df=df, prefix="train", suffix=x_objective))
        y[0].extend(get_column(df=df, prefix="train", suffix=y_objective))
        if has_inner_cv:
            x[1].extend(get_column(df=df, prefix="inner_cv", suffix=x_objective))
            y[1].extend(get_column(df=df, prefix="inner_cv", suffix=y_objective))
        x[2].extend(get_column(df=df, prefix="test", suffix=x_objective))
        y[2].extend(get_column(df=df, prefix="test", suffix=y_objective))
    if x_label is None:
        x_label = x_objective
    if y_label is None:
        y_label = y_objective
    multiclass_scatter_to_ax(
        ax=ax, x=x, y=y,
        x_label=x_label, y_label=y_label, class_labels=class_labels,
        legend_loc='lower right',
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, alpha=alpha,
        interpolate=interpolate, add_ellipses=add_ellipses)


def algorithm_folds_scatter_plot_save(fold_dfs: Sequence[DataFrame], x_objective: str, y_objective: str,
                                      x_min: float, y_min: float,
                                      x_max: float, y_max: float,
                                      save_file: str,
                                      x_label: str = None, y_label: str = None,
                                      interpolate: bool = True, add_ellipses: bool = ADD_ELLIPSES_DEFAULT):
    fig, ax = plt.subplots()
    algorithm_folds_scatter_plot(ax=ax, fold_dfs=fold_dfs, x_objective=x_objective, y_objective=y_objective,
                                 x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                 x_label=x_label, y_label=y_label, interpolate=interpolate, add_ellipses=add_ellipses)
    smart_save_fig(path=save_file)


def folds_scatter_plots_from_saved_hof_one_alg_to_ax(
        saved_hof: SavedHoF, ax,
        x_objective: str, y_objective: str,
        x_min: Optional[float] = None, y_min: Optional[float] = None,
        x_max: Optional[float] = None, y_max: Optional[float] = None,
        alpha: Optional[float] = None,
        x_label: str = None, y_label: str = None):
    f = saved_hof.path()
    if os.path.isdir(f):
        fold_dfs = FITNESS.read_fold_dfs(hof_dir=f)
        if len(fold_dfs) > 0:
            algorithm_folds_scatter_plot(
                ax=ax,
                fold_dfs=fold_dfs, x_objective=x_objective, y_objective=y_objective,
                x_min=x_min, y_min=y_min,
                x_max=x_max, y_max=y_max,
                alpha=alpha,
                x_label=x_label, y_label=y_label)
        else:
            print("Unable to create dataframes from directory " + str(f))
    else:
        print("path is not a directory: " + str(f))


def folds_scatter_plots_from_saved_hof_one_alg_save(
        saved_hof: SavedHoF, save_path: str,
        x_objective: str, y_objective: str,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        x_label: str = None, y_label: str = None):
    fig, ax = plt.subplots()
    folds_scatter_plots_from_saved_hof_one_alg_to_ax(
        saved_hof=saved_hof,
        ax=ax, x_objective=x_objective, y_objective=y_objective,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        x_label=x_label, y_label=y_label)
    alg_name = saved_hof.name()
    algo_save_file = os.path.join(save_path, alg_name + ".png")
    smart_save_fig(path=algo_save_file)


def folds_scatter_plots_from_saved_hofs(saved_hofs: Sequence[SavedHoF], save_path: str,
                                        x_objective: str, y_objective: str,
                                        x_min: float, y_min: float,
                                        x_max: float, y_max: float,
                                        x_label: str = None, y_label: str = None):
    for hof in saved_hofs:
        folds_scatter_plots_from_saved_hof_one_alg_save(
            saved_hof=hof, save_path=save_path, x_objective=x_objective, y_objective=y_objective,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            x_label=x_label, y_label=y_label)
