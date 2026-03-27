from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pandas import DataFrame
from sklearn.decomposition import IncrementalPCA

from consts import FONT_SIZE
from plots.plot_consts import PRINCIPAL_COMPONENT_STR
from plots.plot_utils import default_color_list
from util.dataframe.dataframes import col_as_list
from util.printer.printer import Printer, NULL_PRINTER
from util.progress_observer import SmartProgressObserverFactory
from util.table.table import Table
from util.table.table_utils import n_row, n_col


MAX_PCA_CLASSES = 30
DEFAULT_MAX_N_COMPONENTS_FOR_PLOTS = 6
"""Plots will be created that consider from the first two to the first N principal components."""
DEFAULT_MAX_N_COMPONENTS_FOR_TABLES = 20  # 20 are used in some studies to remove population structure...


def drop_na_for_pca(view: Table, outcome: DataFrame, printer: Printer = NULL_PRINTER) -> Tuple[Table, DataFrame]:
    """Drops NaN by dropping whole rows."""
    initial_rows = n_row(view)
    rows_to_keep = view.rows_without_nan()
    view = view.select_rows(selected=rows_to_keep)
    if initial_rows != n_row(view):
        printer.print("While plotting PCA, " + str(initial_rows - n_row(view)) + " samples were dropped due to NaNs.")
    outcome = outcome.iloc[rows_to_keep, :]
    return view, outcome


def impute_na_for_pca(view: Table, printer: Printer = NULL_PRINTER) -> Table:
    n_missing = view.n_missing()
    if n_missing > 0:
        printer.print("While plotting PCA, " + str(n_missing) +
                      " missing values are imputed by feature average (or dropped if a whole column is missing).")
        view = view.impute()
    return view


def pc_str(pc_index: int) -> str:
    """Indices start from 0."""
    return PRINCIPAL_COMPONENT_STR + " " + str(pc_index + 1)


def principal_components(
        view: Table,
        outcome: DataFrame,
        n_components: int,
        impute_nan: bool = True,
        printer: Printer = NULL_PRINTER) -> Tuple[DataFrame, DataFrame]:
    """If not imputing nan, samples with nan values are dropped.
    Standardization and PCA fit and transform are done in chunks to save memory.
    Outcome is returned because it can be modified if whole rows are dropped due to NaN."""
    initial_rows = n_row(view)
    if initial_rows != n_row(outcome):
        raise ValueError(
            "View and outcome dataframes do not have the same number of rows.\n" +
            "View rows: " + str(initial_rows) + "\n" +
            "Outcome rows: " + str(n_row(outcome)) + "\n")
    if impute_nan:
        view = impute_na_for_pca(view=view, printer=printer)
    else:
        view, outcome = drop_na_for_pca(view=view, outcome=outcome, printer=printer)  # PCA does not work with NA
    n_components = min(n_components, n_col(view))
    view = view.standardize()
    pof = SmartProgressObserverFactory(minutes_of_quiet=1, printer=printer)
    po = pof.create_progress_observer(job_name="Fitting PCA")
    po.notify_start()
    ipca = IncrementalPCA(n_components=n_components, batch_size=view.default_chunk_rows())
    processed_rows = 0
    tot_rows = view.n_row()
    for chunk in view.chunks_df():
        ipca.partial_fit(chunk)
        processed_rows += n_row(chunk)
        po.notify_progress(proportion=processed_rows/tot_rows, text="rows " + str(processed_rows) + "/" + str(tot_rows))
    po.notify_end()
    po = pof.create_progress_observer(job_name="Transforming")
    po.notify_start()
    processed_rows = 0
    transformed_chunks = []
    for chunk in view.chunks_df():
        transformed_chunks.append(ipca.transform(chunk))
        processed_rows += n_row(chunk)
        po.notify_progress(proportion=processed_rows / tot_rows,
                           text="rows " + str(processed_rows) + "/" + str(tot_rows))
    po.notify_end()
    principal_components_array = np.vstack(transformed_chunks)
    index = pd.Index(view.rownames())
    principal_df = pd.DataFrame(data=principal_components_array, index=index)
    if n_col(principal_components_array) >= 2:
        principal_df.columns = [pc_str(pc_index=i) for i in range(n_components)]
    else:
        principal_df.columns = [view.colnames()[0]]
    return principal_df, outcome


def pca2d_from_components(
        ax: Axes,
        principal_component_a: Sequence[float],
        principal_component_b: Sequence[float],
        pc_a_name: str,
        pc_b_name: str,
        outcome: Sequence,
        show_counts: bool = True,
        font_size: int = FONT_SIZE,
        order_by_counts: bool = True,
        point_size: float = 50,
        legend_loc: str = 'best',
        printer: Printer = NULL_PRINTER):
    if len(principal_component_a) != len(principal_component_b) or len(principal_component_a) != len(outcome):
        raise ValueError("Components and outcome must all have the same length.")
    counter = collections.Counter(outcome).most_common()
    targets = [c[0] for c in counter]
    if not order_by_counts:
        targets.sort()
        counter.sort()
    n_targets = len(targets)
    colors = default_color_list(n_colors=n_targets, invert=False)
    n_colors = len(colors)
    if n_targets <= n_colors and n_targets <= MAX_PCA_CLASSES:
        final_df = DataFrame(data={pc_a_name:principal_component_a, pc_b_name:principal_component_b, "outcome":outcome})
        printer.print("Creating plot")
        with plt.style.context({'font.size': font_size}):
            ax.set_xlabel(pc_a_name, fontsize=font_size)
            ax.set_ylabel(pc_b_name, fontsize=font_size)
            for target, color in zip(targets, colors):
                indices_to_keep = [o == target for o in outcome]
                x = final_df.loc[indices_to_keep, pc_a_name]
                y = final_df.loc[indices_to_keep, pc_b_name]
                ax.scatter(x, y, color=color, s=point_size)
                # setting warnings.catch_warnings() does not work for ignoring the color warnings.
            if show_counts:
                legend_entries = [str(c[0]) + " (" + str(c[1]) + ")" for c in counter]
            else:
                legend_entries = targets
            ax.legend(legend_entries, loc=legend_loc)
            ax.grid()
        printer.print("Plot created")
    else:
        raise ValueError("Too many outcome classes.")


def pca2d_view_ax(ax: Axes,
                  view: Table,
                  outcome: DataFrame,
                  show_counts: bool = True,
                  font_size: int = FONT_SIZE,
                  order_by_counts: bool = True,
                  point_size: float = 50,
                  legend_loc: str = 'best',
                  impute_nan: bool = True,
                  principal_component_a: int = 0,
                  principal_component_b: int = 1,
                  printer: Printer = NULL_PRINTER):
    """If not imputing nan, samples with nan values are dropped.
    Standardization and PCA fit and transform are done in chunks to save memory."""
    initial_rows = n_row(view)
    if initial_rows != n_row(outcome):
        raise ValueError(
            "View and outcome dataframes do not have the same number of rows.\n" +
            "View rows: " + str(initial_rows) + "\n" +
            "Outcome rows: " + str(n_row(outcome)) + "\n")
    if len(outcome.columns) == 1:
        if n_col(view) >= 2:
            n_components = max(2, max(principal_component_a, principal_component_b) + 1)
        else:
            n_components = 1
        principal_df, outcome = principal_components(
            view=view,
            outcome=outcome,
            n_components=n_components,
            impute_nan=impute_nan,
            printer=printer)
        counter = collections.Counter(outcome.iloc[:, 0]).most_common()
        targets = [c[0] for c in counter]
        if not order_by_counts:
            targets.sort()
            counter.sort()
        n_targets = len(targets)
        colors = default_color_list(n_colors=n_targets, invert=False)
        n_colors = len(colors)
        if n_targets <= n_colors and n_targets <= MAX_PCA_CLASSES:
            outcome_list = outcome.iloc[:, 0].astype("category").cat.codes.tolist()
            if n_col(principal_df) >= 2:
                pc1_str = pc_str(pc_index=principal_component_a)
                pc2_str = pc_str(pc_index=principal_component_b)
                principal_component_a_list = col_as_list(df=principal_df, col=principal_component_a)
                principal_component_b_list = col_as_list(df=principal_df, col=principal_component_b)
                ax1_str = pc1_str
                ax2_str = pc2_str
            else:
                ax1_str = view.colnames()[0]
                ax2_str = "label"
                principal_component_a_list = col_as_list(df=principal_df, col=0)
                principal_component_b_list = outcome_list
            pca2d_from_components(
                ax=ax,
                principal_component_a=principal_component_a_list,
                principal_component_b=principal_component_b_list,
                pc_a_name=ax1_str,
                pc_b_name=ax2_str,
                outcome=outcome_list,
                show_counts=show_counts,
                font_size=font_size,
                order_by_counts=order_by_counts,
                point_size=point_size,
                legend_loc=legend_loc,
                printer=printer)
        else:
            raise ValueError("Too many outcome classes.")
    else:
        raise ValueError("Outcome has not exactly 1 column.")
