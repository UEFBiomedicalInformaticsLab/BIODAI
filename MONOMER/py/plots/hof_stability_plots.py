import os

import pandas as pd
from pandas import DataFrame

from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import \
    stability_by_weights_from_counts, stability_by_unions_from_counts
from plots.plot_utils import single_line_plot_to_path
from plots.hof_utils import hof_used_features_fold_dfs
from util.components_transform import HigherKComponents
from util.dataframes import sum_by_columns
from util.list_math import mean_all_vs_others, n_true_in_common, jaccard, dice
from util.sequence_utils import to_common_labels


def stability_weight_top_k_plot(stabilities: [float], plot_path: str):
    single_line_plot_to_path(y=stabilities, plot_path=plot_path, y_label="hall of fame stability between folds",
                             x_label="Number of top features")


def common_features_top_k_plot(n_common_features: [float], plot_path: str):
    single_line_plot_to_path(y=n_common_features, plot_path=plot_path, y_label="Common features between folds",
                             x_label="Number of top features")


def jaccard_top_k_plot(jac: [float], plot_path: str):
    single_line_plot_to_path(y=jac, plot_path=plot_path, y_label="Jaccard between folds",
                             x_label="Number of top features")


def dice_top_k_plot(dice_values: [float], plot_path: str):
    single_line_plot_to_path(y=dice_values, plot_path=plot_path, y_label="Dice coefficient between folds",
                             x_label="Number of top features")


def stability_union_top_k_plot(stability: [float], plot_path: str):
    single_line_plot_to_path(y=stability, plot_path=plot_path, y_label="hall of fame stability between folds",
                             x_label="Number of top features")


def smaller_feature_set_size(fold_counts: [[int]]) -> int:
    return min([sum([a != 0 for a in fc]) for fc in fold_counts])


def dfs_to_counts(dfs: [DataFrame]) -> [[int]]:
    return [sum_by_columns(df) for df in dfs]


def mean_common_features(fold_counts: [[int]]) -> float:
    return mean_all_vs_others(elems=fold_counts, measure_function=n_true_in_common)


def mean_jaccard(fold_counts: [[int]]) -> float:
    return mean_all_vs_others(elems=fold_counts, measure_function=jaccard)


def mean_dice(fold_counts: [[int]]) -> float:
    return mean_all_vs_others(elems=fold_counts, measure_function=dice)


def dfs_to_hof_stability_plots(dfs: [DataFrame], plot_dir: str):
    if len(dfs) > 1:
        fold_counts = dfs_to_counts(dfs)
        fold_counts = to_common_labels(fold_counts)
        fold_counts = [pd.Series(li).fillna(0).tolist() for li in fold_counts]
        max_k = smaller_feature_set_size(fold_counts=fold_counts)
        stabilities = []
        n_common_features = []
        jac = []
        dice_values = []
        stabilities_union = []
        for k in range(1, max_k+1):
            components_transform = HigherKComponents(k)
            fold_counts_k = [components_transform.apply(c) for c in fold_counts]
            stabilities.append(stability_by_weights_from_counts(counts=fold_counts_k))
            n_common_features.append(mean_common_features(fold_counts_k))
            jac.append(mean_jaccard(fold_counts_k))
            dice_values.append(mean_dice(fold_counts_k))
            stabilities_union.append(stability_by_unions_from_counts(counts=fold_counts_k))
        stability_weight_top_k_plot(stabilities=stabilities, plot_path=plot_dir + "/hof_weight_stability.png")
        # stability_union_top_k_plot(stability=stabilities_union, plot_path=plot_dir + "/hof_union_stability.png")
        # Union top k is the same as Dice.
        common_features_top_k_plot(n_common_features=n_common_features, plot_path=plot_dir+"/common_features.png")
        jaccard_top_k_plot(jac=jac, plot_path=plot_dir + "/jaccard.png")
        dice_top_k_plot(dice_values=dice_values, plot_path=plot_dir + "/dice.png")  # It is the same as the union top k.


def hof_stability_plot_for_directory(hof_dir: str):
    dats = hof_used_features_fold_dfs(hof_dir=hof_dir)
    dfs_to_hof_stability_plots(dfs=dats, plot_dir=hof_dir)


def hof_stability_plots_for_main_directory(main_hofs_dir: str):
    if os.path.isdir(main_hofs_dir):
        subfolders = [f.path for f in os.scandir(main_hofs_dir) if f.is_dir()]
        for f in subfolders:
            hof_stability_plot_for_directory(hof_dir=f)
