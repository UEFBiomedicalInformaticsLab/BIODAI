from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

from consts import FONT_SIZE
from input_data.input_data import InputData
from input_data.input_data_saver import smart_save_df_unchecked
from load_omics_views import MRNA_NAME
from plots.pca import principal_components, DEFAULT_MAX_N_COMPONENTS_FOR_PLOTS, MAX_PCA_CLASSES, \
    DEFAULT_MAX_N_COMPONENTS_FOR_TABLES
from plots.plot_consts import DEFAULT_PALETTE_NAME
from plots.plot_utils import smart_save_fig
from plots.plotter.pca_plotter import pca_component_combinations_from_components, PCA_PLOTTER_SHOW_COUNTS
from plots.show_views import odds_ratio_horizontal_df, MAX_FEATURES_FOR_GRIDS
from plots.subplots import subplots
from util.printer.printer import Printer, NULL_PRINTER, UNBUFFERED_OUT_PRINTER
from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_utils import n_col
from util.utils import IllegalStateError
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from itertools import combinations


def extract_outcome_data(input_data: InputData) -> DataFrame:
    for outcome in input_data.outcomes():
        outcome_d = outcome.data()
        if len(outcome_d.columns) == 1:
            if len(set(outcome_d.iloc[:, 0])) <= 20:
                return outcome_d
    raise IllegalStateError()


def pca_subplots_from_components_one_dataset(
        principal_df: DataFrame,
        outcome_seq: Sequence,
        save_path: str,
        row_names: Optional[Sequence[str]] = None,
        n_components: int = 2,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        font_size: int = FONT_SIZE,
        show_counts: bool = PCA_PLOTTER_SHOW_COUNTS):
    pca_subplots_from_components_multiple_datasets(
        principal_dfs=[principal_df],
        outcome_seqs=[outcome_seq],
        save_path=save_path,
        row_names=row_names,
        n_components=n_components,
        n_cols=n_cols,
        share_axes=share_axes,
        font_size=font_size,
        show_counts=show_counts
    )


def pca_subplots_from_components_multiple_datasets(
        principal_dfs: Sequence[DataFrame],
        outcome_seqs: Sequence[Sequence],
        save_path: str,
        row_names: Optional[Sequence[str]] = None,
        n_components: int = 2,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        font_size: int = FONT_SIZE,
        show_counts: bool = PCA_PLOTTER_SHOW_COUNTS):
    plots_per_input = ((n_components*n_components) - n_components) // 2
    if n_cols is None:
        if len(principal_dfs) > 1:
            n_cols = plots_per_input
            # If there are multiple datasets and n_cols is not specified we put one dataset per row.
    if share_axes:
        sharex = True
        sharey = True
    elif n_cols == plots_per_input:
        sharex = "col"
        sharey = "col"
    else:
        sharex = False
        sharey = False
    plotters = []
    for principal_df, outcome_seq in zip(principal_dfs, outcome_seqs):
        plotters.extend(
            pca_component_combinations_from_components(
                principal_df=principal_df, outcome_seq=outcome_seq, n_components=n_components, show_counts=show_counts))
    subplots(plotters=plotters, save_path=save_path, ncols=n_cols,
             row_names=row_names, sharex=sharex, sharey=sharey, font_size=font_size)


def pca_subplots(
        input_data: Sequence[InputData],
        save_path: str,
        view_name: str = MRNA_NAME,
        row_names: Optional[Sequence[str]] = None,
        n_components: int = 2,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        printer: Printer = NULL_PRINTER):
    principal_dfs = []
    outcome_seqs = []
    for i in input_data:
        principal_df, outcome = principal_components(
            view=i.view(view_name=view_name),
            outcome=extract_outcome_data(input_data=i),
            n_components=n_components,
            printer=printer)
        principal_dfs.append(principal_df)
        outcome_seqs.append(outcome.iloc[:, 0].astype("category").cat.codes.tolist())
    pca_subplots_from_components_multiple_datasets(
        principal_dfs=principal_dfs,
        outcome_seqs=outcome_seqs,
        save_path=save_path,
        row_names=row_names,
        n_components=n_components,
        n_cols=n_cols,
        share_axes=share_axes)


def pca_subplots_range(
        input_data: Sequence[InputData],
        save_dir: str,
        view_name: str = MRNA_NAME,
        row_names: Optional[Sequence[str]] = None,
        max_n_components: int = 5,
        n_cols: Optional[int] = None,
        printer: Printer = NULL_PRINTER):
    principal_dfs = []
    outcome_seqs = []
    for i in input_data:
        principal_df, outcome = principal_components(
            view=i.view(view_name=view_name),
            outcome=extract_outcome_data(input_data=i),
            n_components=max_n_components,
            printer=printer)
        principal_dfs.append(principal_df)
        outcome_seqs.append(outcome.iloc[:, 0].astype("category").cat.codes.tolist())
    for i in range(2, max_n_components+1):
        save_path = os.path.join(save_dir, "pca_" + str(i))
        pca_subplots_from_components_multiple_datasets(
            principal_dfs=principal_dfs, outcome_seqs=outcome_seqs, save_path=save_path,
            row_names=row_names, n_components=i, n_cols=n_cols)


def pca2d_view(view: Table,
               view_name: str,
               outcome: DataFrame,
               outcome_name: str,
               directory: str,
               max_n_components_for_plots: int = DEFAULT_MAX_N_COMPONENTS_FOR_PLOTS,
               printer: Printer = UNBUFFERED_OUT_PRINTER,
               show_counts: bool = True,
               font_size: int = FONT_SIZE) -> DataFrame:
    principal_df, outcome = principal_components(
        view=view,
        outcome=outcome,
        n_components=DEFAULT_MAX_N_COMPONENTS_FOR_TABLES,
        printer=printer)
    outcome_seq = outcome.iloc[:, 0].astype("category").tolist()
    if len(set(outcome_seq)) == 2:  # Binary outcome
        printer.print(f"Plotting horizontal odds ratio for principal components and outcome {outcome_name}")
        odds_ratio_horizontal_df(
            view_df=principal_df, name=view_name+"_principal_components", outcome_df=outcome, outcome_name=outcome_name,
            directory=directory, printer=printer)
    for i in range(2, max_n_components_for_plots + 1):
        printer.print("Creating plot for the first " + str(i) + " components.")
        try:
            save_path = os.path.join(directory, view_name + "_" + outcome_name + "_pca_" + str(i))
            pca_subplots_from_components_one_dataset(
                principal_df=principal_df, outcome_seq=outcome_seq, save_path=save_path,
                n_components=i, font_size=font_size, show_counts=show_counts)
        except ValueError as e:
            printer.print("PCA not plotted for view " + str(view_name) + ", outcome " + str(outcome_name) +
                          " and " + str(i) + " components." + "\n" +
                          str(e))
    printer.print("Writing the principal components to file.")
    smart_save_df_unchecked(directory=directory, table_name=view_name + "_" + outcome_name + "_pca", df=principal_df)
    return principal_df


def plot_pc_view_correlations(principal_df: DataFrame, view: Table,
                              principal_name: str,
                              view_name: str,
                              directory: str,
                              printer: Printer = UNBUFFERED_OUT_PRINTER) -> DataFrame:
    """
    Computes the absolute Kendall correlation between each principal component and each feature.
    Plots a heatmap and saves it to a PNG file.

    Parameters:
    - principal_df: DataFrame with principal components (columns = components)
    - view: Table with features (columns = features)
    - output_path: Path to save the heatmap PNG

    Returns:
    - kendall_matrix: DataFrame of absolute Kendall correlation scores
    """
    printer.title_print("Plot principal component - feature correlations")
    view = view.select_rows_by_names(names=principal_df.index)
    view_df = view.to_dataframe()

    # Check for missing data in features
    missing_counts = view_df.isnull().sum()
    if (missing_counts > 0).any():
        printer.print("Missing values per feature:")
        printer.print(missing_counts[missing_counts > 0])
    else:
        printer.print("No missing values detected in the features.")

    kendall_matrix = pd.DataFrame(index=principal_df.columns, columns=view_df.columns)

    for pc in principal_df.columns:
        for feature in view_df.columns:
            # Drop missing values pairwise
            valid_data = pd.concat([principal_df[pc], view_df[feature]], axis=1).dropna()
            if valid_data.empty:
                kendall_matrix.loc[pc, feature] = np.nan
                continue

            tau, _ = kendalltau(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
            kendall_matrix.loc[pc, feature] = abs(tau)

    kendall_matrix = kendall_matrix.astype(float)

    # Optionally drop columns with all NaNs
    kendall_matrix.dropna(axis=1, how='all', inplace=True)

    # Create a mask for missing values
    mask = kendall_matrix.isnull()

    plt.figure(figsize=(12, 8))
    sns.heatmap(kendall_matrix, annot=True, cmap="viridis",
                cbar_kws={'label': 'Absolute Kendall Correlation'},
                mask=mask)
    plt.title("Absolute Kendall Correlation: Component vs Feature")
    plt.xlabel("Feature")
    plt.ylabel("Principal Component")
    plt.tight_layout()

    output_path = os.path.join(directory, f"{principal_name}_component_{view_name}_feature_kendall_heatmap.png")
    printer.print(f"Saving figure to {output_path}")
    smart_save_fig(path=output_path, printer=printer)

    return kendall_matrix


def plot_top_pc_feature_combinations(
        kendall_matrix: DataFrame, principal_df: DataFrame, view_df: DataFrame,
        principal_name: str,
        view_name: str,
        directory: str,
        printer: Printer = UNBUFFERED_OUT_PRINTER,
        palette: str = DEFAULT_PALETTE_NAME):

    printer.title_print("Plot top principal components - feature combinations")
    if n_col(principal_df) < 2:
        printer.title_print("Not plotting because there are less than 2 principal components.")
        return

    # Flatten the correlation matrix to get all combinations of PC and feature
    correlations = []
    for pc in kendall_matrix.index:
        for feature in kendall_matrix.columns:
            value = kendall_matrix.loc[pc, feature]
            if not np.isnan(value):
                correlations.append((pc, feature, value))

    # Generate all combinations of 2 PCs and 1 feature
    pc_combinations = list(combinations(kendall_matrix.index, 2))
    feature_list = kendall_matrix.columns.tolist()

    combo_scores = []
    for pc1, pc2 in pc_combinations:
        for feature in feature_list:
            score = kendall_matrix.loc[pc1, feature] + kendall_matrix.loc[pc2, feature]
            combo_scores.append(((pc1, pc2, feature), score))

    # Sort combinations by score and select top 16
    top_combos = sorted(combo_scores, key=lambda x: x[1], reverse=True)[:16]

    # Create subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    # Get the categorical palette colors
    categorical_colors = sns.color_palette(palette)
    if len(categorical_colors) < 2:
        raise ValueError("Palette must contain at least two colors for continuous gradient.")

    # Create a custom colormap from the first two colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", categorical_colors[:2])

    for i, ((pc1, pc2, feature), _) in enumerate(top_combos):
        ax = axes[i]
        data = pd.concat([principal_df[[pc1, pc2]], view_df[[feature]]], axis=1).dropna()
        ax.set_xlabel(pc1)
        ax.set_ylabel(pc2)

        if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > 10:
            # Use custom gradient for continuous features
            scatter = ax.scatter(data[pc1], data[pc2], c=data[feature], cmap=custom_cmap)
            plt.colorbar(scatter, ax=ax)
        else:
            # Use seaborn palette for categorical features and show legend
            sns.scatterplot(x=pc1, y=pc2, hue=feature, data=data, ax=ax, palette=palette)
            ax.legend(title=None, loc='best')

        ax.set_title(f"Colored by {feature}")

    plt.tight_layout()
    output_path = os.path.join(directory, f"{principal_name}_component_{view_name}_top_pc_feature_combinations.png")
    printer.print(f"Saving figure to {output_path}")
    smart_save_fig(path=output_path, printer=printer)


def pca2d(data: InputData, directory: str, printer: Printer = UNBUFFERED_OUT_PRINTER, skip_huge_views: bool = True):
    for outcome in data.outcomes():
        outcome_data = outcome.data()
        if len(outcome_data.columns) == 1:
            if len(set(outcome_data.iloc[:, 0])) <= MAX_PCA_CLASSES:  # Perhaps check is not needed at this point.
                outcome_name = outcome.name()
                for view_name in data.views_dict():
                    view_data = data.view(view_name)
                    if skip_huge_views and view_data.size() > DEFAULT_MAX_CACHEABLE_CELLS:
                        printer.print("Skipping PCA for huge view " + str(view_name))
                    else:
                        printer.print("Creating PCA for view " + str(view_name))
                        principal_df = pca2d_view(
                            view=view_data, view_name=view_name, outcome=outcome_data, outcome_name=outcome_name,
                            directory=directory, printer=printer)
                        for feature_view_name in data.views_dict():
                            feature_view = data.view(feature_view_name)
                            # if vn != view_name and data.view(vn).n_col() <= MAX_FEATURES_FOR_GRIDS:
                            if feature_view.n_col() <= MAX_FEATURES_FOR_GRIDS:
                                kendall_matrix = plot_pc_view_correlations(
                                    principal_df=principal_df, view=feature_view,
                                    principal_name=view_name, view_name=feature_view_name, directory=directory,
                                    printer=printer)
                                plot_top_pc_feature_combinations(
                                    kendall_matrix=kendall_matrix, principal_df=principal_df,
                                    view_df=feature_view.to_dataframe(),
                                    principal_name=view_name,
                                    view_name=feature_view_name,
                                    directory=directory, printer=printer)
