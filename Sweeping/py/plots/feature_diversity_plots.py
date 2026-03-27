import os
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from consts import FEATURE_COUNTS_EXTENSION, FEATURE_COUNTS_PREFIX, GEN_STR
from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import \
    stability_by_weights_from_counts, stability_by_unions_from_counts
from plots.hof_stability_plots import mean_common_features
from util.components_transform import ComponentsTransform, IdentityComponentsTransform, HigherKComponents
from util.math.list_math import vector_mean
import seaborn as sns

from util.sequence_utils import stable_uniques


def merge_df(dfs: list[DataFrame], sort=False) -> DataFrame:
    """Merges by concatenating by rows."""
    merged_df = pd.concat(dfs, axis=0, ignore_index=True, join='outer', sort=sort)
    merged_df = merged_df.fillna(0)
    return merged_df


def stability_in_time(df):
    stability = []
    for r in range(0, len(df)-1):
        row1 = df.iloc[[r]].values.flatten().tolist()
        row2 = df.iloc[[r+1]].values.flatten().tolist()
        stability.append(stability_by_weights_from_counts([row1, row2]))
    return stability


def dfs_to_stability_in_time_plot(dfs: list, plot_dir: str):
    stabilities = []
    for df in dfs:
        stabilities.append(stability_in_time(df))
    if len(stabilities) > 0 and GEN_STR in dfs[0].columns:
        gens = dfs[0][GEN_STR][1:len(dfs[0])]
        if len(gens) > 1:  # We want at least a segment to plot.
            stability = vector_mean(stabilities)
            stability_plot(gen=gens, stability=stability, stability_label="stability in time",
                           plot_path=os.path.join(plot_dir, "stability_in_time.png"))
            for s_i in range(0, len(stabilities)):
                stability = stabilities[s_i]
                stability_plot(gen=gens, stability=stability, stability_label="stability in time",
                               plot_path=os.path.join(plot_dir, "stability_in_time_fold_" + str(s_i) + ".png"))


def dfs_to_stability_between_folds_plot(dfs: list, plot_dir: str,
                                        components_transform: ComponentsTransform = IdentityComponentsTransform()):
    if len(dfs) > 1 and GEN_STR in dfs[0].columns:  # We need at least two folds to compare and the generation column.
        gens = []
        stability = []
        merged_df = merge_df(dfs=dfs)
        for row in range(len(dfs[0])):
            gen = dfs[0][GEN_STR][row]
            gens.append(gen)
            row_counts = merged_df.loc[merged_df[GEN_STR] == gen, merged_df.columns != GEN_STR]
            stability.append(
                stability_by_weights_from_counts(row_counts.values.tolist(),
                                                 components_transform=components_transform))
        stability_plot(gen=gens, stability=stability, stability_label="stability of folds",
                       plot_path=os.path.join(plot_dir, "stability_between_folds.png"))


def stability_of_weights_between_folds_top_k(merged_df: DataFrame, k: int) -> list[float]:
    gens = stable_uniques(merged_df[GEN_STR])
    stability = []
    for row in range(len(gens)):
        gen = gens[row]
        row_counts = merged_df.loc[merged_df[GEN_STR] == gen, merged_df.columns != GEN_STR]
        stability.append(
            stability_by_weights_from_counts(row_counts.values.tolist(), components_transform=HigherKComponents(k)))
    return stability


def stability_of_unions_between_folds_top_k(merged_df: DataFrame, k: int) -> list[float]:
    gens = stable_uniques(merged_df[GEN_STR])
    stability = []
    for row in range(len(gens)):
        gen = gens[row]
        row_counts = merged_df.loc[merged_df[GEN_STR] == gen, merged_df.columns != GEN_STR]
        stability.append(
            stability_by_unions_from_counts(row_counts.values.tolist(), components_transform=HigherKComponents(k)))
    return stability


def common_features_between_folds_top_k(merged_df: DataFrame, k: int) -> list[float]:
    gens = stable_uniques(merged_df[GEN_STR])
    stability = []
    for row in range(len(gens)):
        gen = gens[row]
        row_counts = merged_df.loc[merged_df[GEN_STR] == gen, merged_df.columns != GEN_STR]
        stability.append(
            mean_common_features([HigherKComponents(k).apply(li) for li in row_counts.values.tolist()]))
    return stability


def stability_top_k_df_to_plot(plot_dir: str, plot_name: str, stability_df: DataFrame):
    plt.subplots(figsize=(20, 15))
    sns.heatmap(stability_df, annot=True)
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=600)
    plt.close()


def dfs_to_stability_between_folds_top_k_plots(dfs: list, plot_dir: str):
    k_values = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    if len(dfs) > 1 and GEN_STR in dfs[0].columns:  # We need at least two folds to compare.
        merged_df = merge_df(dfs=dfs)
        gens = dfs[0][GEN_STR]
        stability_cols_weights = dict.fromkeys(k_values)
        stability_cols_unions = dict.fromkeys(k_values)
        union_size_cols = dict.fromkeys(k_values)
        for k in k_values:
            stability_cols_weights[k] = stability_of_weights_between_folds_top_k(merged_df=merged_df, k=k)
            stability_cols_unions[k] = stability_of_unions_between_folds_top_k(merged_df=merged_df, k=k)
            union_size_cols[k] = common_features_between_folds_top_k(merged_df=merged_df, k=k)
        stability_df_weights = DataFrame.from_dict(stability_cols_weights)
        stability_df_weights.index = gens
        stability_top_k_df_to_plot(
            plot_dir=plot_dir, plot_name="stability_of_weights_between_folds_top_k.png",
            stability_df=stability_df_weights)
        stability_df_unions = DataFrame.from_dict(stability_cols_unions)
        stability_df_unions.index = gens
        stability_top_k_df_to_plot(
            plot_dir=plot_dir, plot_name="stability_of_unions_between_folds_top_k.png",
            stability_df=stability_df_unions)
        stability_df_common = DataFrame.from_dict(union_size_cols)
        stability_df_common.index = gens
        stability_top_k_df_to_plot(
            plot_dir=plot_dir, plot_name="common_features_between_folds_top_k.png",
            stability_df=stability_df_common)


def stability_plot(gen, stability, stability_label: str, plot_path: str):
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    lines = []
    lines = lines + ax.plot(gen, stability, "b-", label=stability_label)

    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc="center right")

    plt.savefig(plot_path, bbox_inches='tight', dpi=600)
    plt.close()


def dfs_to_feature_diversity_plots(dfs: list, plot_dir: str):
    dfs_to_stability_between_folds_plot(dfs=dfs, plot_dir=plot_dir)


def load_saved_feature_lists(direct: str) -> list[DataFrame]:
    """Fold files are read in alphabetical order.
    Returns empty list if there is nothing to read. Tries to read feature counts files."""
    dats = []
    files = os.listdir(direct)
    files.sort()
    for file in files:
        if file.startswith(FEATURE_COUNTS_PREFIX) and file.endswith("." + FEATURE_COUNTS_EXTENSION):
            dats.append(pd.read_csv(filepath_or_buffer=os.path.join(direct, file)))
    return dats


def feature_diversity_plots_for_directory(direct: str):
    dats = load_saved_feature_lists(direct=direct)
    if len(dats) > 0:
        dfs_to_feature_diversity_plots(dfs=dats, plot_dir=direct)
    dfs_to_stability_in_time_plot(dfs=dats, plot_dir=direct)
    dfs_to_stability_between_folds_top_k_plots(dfs=dats, plot_dir=direct)
