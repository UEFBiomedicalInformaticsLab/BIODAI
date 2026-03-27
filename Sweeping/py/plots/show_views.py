import pathlib
import warnings

import seaborn as sns
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections.abc import Sequence
from pandas import concat, DataFrame
from pandas.errors import InvalidIndexError
from seaborn import color_palette, boxplot
from input_data.input_data import InputData
from input_data.outcome import Outcome
from plots.plot_consts import DEFAULT_PALETTE_NAME
from plots.plot_utils import smart_save_fig
from util.printer.printer import Printer, OutPrinter
from util.progress_observer import SmartProgressObserverFactory, ProgressObserverFactory, \
    DEFAULT_PROGRESS_OBSERVER_FACTORY
from util.sequence_utils import safe_nanmin, safe_nanmax
from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.iterable_utils import SizedIterable
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from statsmodels.api import Logit, add_constant
from statsmodels.stats.multitest import fdrcorrection


MAX_FEATURES_FOR_GRIDS = 30


def boxplot_grid(view: Table, name: str, outcomes: Sequence[Outcome],
                 directory: str, printer: Printer = OutPrinter(), max_features: int = MAX_FEATURES_FOR_GRIDS,
                 palette_name: str = DEFAULT_PALETTE_NAME, font_size: int = 14):
    if view.n_col() <= max_features:
        view_df = view.to_dataframe()
        for o in outcomes:
            if o.is_categorical():
                printer.print(f"Plotting box plot grid for view {name} and outcome {o.name()}")
                o_df = o.data()
                try:
                    combined_df = concat([view_df.reset_index(drop=True), o_df.reset_index(drop=True)], axis=1)
                except InvalidIndexError as e:
                    printer.print("Error while concatenating the view and the objective dataframes.")
                    printer.print(str(e))
                    printer.print("Plotting is aborted and the program will continue.")
                    return

                class_col = o.name()
                features = view_df.columns
                n_features = len(features)
                n_cols = 5
                n_rows = math.ceil(n_features / n_cols)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
                axes = axes.flatten()

                n_classes = combined_df[class_col].nunique()
                palette = color_palette(palette_name, n_colors=n_classes)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)

                    for i, feature in enumerate(features):
                        boxplot(
                            x=class_col,
                            y=feature,
                            data=combined_df,
                            ax=axes[i],
                            hue=class_col,
                            palette=palette,
                            dodge=False,
                            width=0.5,
                            showmeans=True,
                            meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=5)
                        )
                        axes[i].set_xlabel(class_col, fontsize=font_size)
                        axes[i].set_ylabel(feature, fontsize=font_size)
                        axes[i].tick_params(axis='x', labelrotation=0, labelsize=font_size)
                        axes[i].tick_params(axis='y', labelsize=font_size)
                        for label in axes[i].get_xticklabels():
                            label.set_ha('center')

                        legend = axes[i].get_legend()
                        if legend:
                            legend.remove()

                for j in range(n_features, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                filename = f"{name}_{class_col}_grid_boxplot.png"
                printer.print(f"Saving plot to {filename}")
                smart_save_fig(path=os.path.join(directory, filename), printer=printer)



def odds_ratio_horizontal_df(view_df: DataFrame, name: str, outcome_df: DataFrame, outcome_name: str,
                             directory: str,
                             printer: Printer = OutPrinter(),
                             palette_name: str = DEFAULT_PALETTE_NAME):
    """Works only for binary outcomes."""
    palette = sns.color_palette(palette_name)
    sig_color = palette[1]
    non_sig_color = palette[0]

    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    processed_df = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(view_df)), columns=view_df.columns)

    # Remove near-constant features
    selector = VarianceThreshold(threshold=1e-5)
    processed_df = pd.DataFrame(selector.fit_transform(processed_df),
                                columns=view_df.columns[selector.get_support()])

    # Check for NaNs
    if processed_df.isnull().any().any():
        printer.print("NaNs remain in processed_df after imputation.")
        return

    combined_df = pd.concat(
        [processed_df.reset_index(drop=True), outcome_df.reset_index(drop=True, inplace=False)], axis=1)

    p_values = []
    valid_features = []
    results = {}

    for feature in processed_df.columns:
        try:
            y_raw = combined_df[outcome_name]
            unique_vals = y_raw.dropna().unique()
            if len(unique_vals) != 2:
                printer.print(f"Skipping {feature}: outcome is not binary.")
                continue
            if not set(unique_vals).issubset({0, 1}):
                mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                y = y_raw.map(mapping).astype(int)
            else:
                y = y_raw.astype(int)
            X = add_constant(combined_df[[feature]])
            model = Logit(y, X).fit(disp=0)
            p_values.append(model.pvalues[feature])
            valid_features.append(feature)
            results[feature] = model
        except Exception as e:
            printer.print(f"Error with feature '{feature}': {e}")
            continue
    if not p_values:
        printer.print("No valid features for logistic regression.")
        return

    _, fdr_corrected = fdrcorrection(p_values)
    fdr_dict = dict(zip(valid_features, fdr_corrected))

    odds_ratios = []
    errors = []
    colors = []
    labels = []
    fdr_values = []

    for feature in valid_features:
        model = results[feature]
        odds_ratio = np.exp(model.params[feature])
        conf = np.exp(model.conf_int().loc[feature])
        fdr = fdr_dict[feature]
        odds_ratios.append(odds_ratio)
        errors.append([odds_ratio - conf[0], conf[1] - odds_ratio])
        colors.append(sig_color if fdr < 0.05 else non_sig_color)
        labels.append(feature)
        fdr_values.append(fdr)

    plot_df = pd.DataFrame({
        'feature': labels,
        'odds_ratio': odds_ratios,
        'error_low': [e[0] for e in errors],
        'error_high': [e[1] for e in errors],
        'color': colors,
        'fdr': fdr_values
    })

    plot_df.sort_values(by=['fdr', 'feature'], inplace=True)
    plot_df = plot_df[::-1]

    labels = plot_df['feature'].tolist()
    odds_ratios = plot_df['odds_ratio'].tolist()
    errors = np.array([plot_df['error_low'].tolist(), plot_df['error_high'].tolist()])
    colors = plot_df['color'].tolist()
    fdr_values = plot_df['fdr'].tolist()

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.6))
    bars = ax.barh(labels, odds_ratios, xerr=errors, color=colors)
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.set_xlabel("Odds Ratio")
    ax.set_title(f"Odds Ratios for {outcome_name} in {name}")

    for i, (bar, fdr) in enumerate(zip(bars, fdr_values)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"FDR={fdr:.3g}", va='center', fontsize=9)

    legend_patches = [
        mpatches.Patch(color=sig_color, label='Significant (FDR < 0.05)'),
        mpatches.Patch(color=non_sig_color, label='Not significant')
    ]
    fig.legend(handles=legend_patches, loc='upper left')
    plt.tight_layout()

    filename = f"{name}_{outcome_name}_horizontal_odds_ratios.png"
    printer.print(f"Saving figure to {os.path.join(directory, filename)}")
    smart_save_fig(path=os.path.join(directory, filename), printer=printer)


def odds_ratio_horizontal_df_old(view_df: DataFrame, name: str, outcome_df: DataFrame, outcome_name: str,
                             directory: str,
                             printer: Printer = OutPrinter(),
                             palette_name: str = DEFAULT_PALETTE_NAME):
    """Works only for binary outcomes."""
    palette = sns.color_palette(palette_name)
    sig_color = palette[1]
    non_sig_color = palette[0]

    features = view_df.columns

    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    processed_df = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(view_df)), columns=view_df.columns)
    combined_df = pd.concat(
        [processed_df.reset_index(drop=True), outcome_df.reset_index(drop=True, inplace=False)], axis=1)

    p_values = []
    valid_features = []
    results = {}

    for feature in features:
        try:
            y_raw = combined_df[outcome_name]
            unique_vals = y_raw.dropna().unique()
            if len(unique_vals) != 2:
                continue
            if not set(unique_vals).issubset({0, 1}):
                mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                y = y_raw.map(mapping).astype(int)
            else:
                y = y_raw.astype(int)
            X = add_constant(combined_df[[feature]])
            model = Logit(y, X).fit(disp=0)
            p_values.append(model.pvalues[feature])
            valid_features.append(feature)
            results[feature] = model
        except:
            continue

    _, fdr_corrected = fdrcorrection(p_values)
    fdr_dict = dict(zip(valid_features, fdr_corrected))

    odds_ratios = []
    errors = []
    colors = []
    labels = []
    fdr_values = []

    for feature in features:
        if feature not in results:
            continue
        model = results[feature]
        odds_ratio = np.exp(model.params[feature])
        conf = np.exp(model.conf_int().loc[feature])
        fdr = fdr_dict[feature]
        odds_ratios.append(odds_ratio)
        errors.append([odds_ratio - conf[0], conf[1] - odds_ratio])
        colors.append(sig_color if fdr < 0.05 else non_sig_color)
        labels.append(feature)
        fdr_values.append(fdr)

    # Create a DataFrame for sorting
    plot_df = pd.DataFrame({
        'feature': labels,
        'odds_ratio': odds_ratios,
        'error_low': [e[0] for e in errors],
        'error_high': [e[1] for e in errors],
        'color': colors,
        'fdr': fdr_values
    })

    # Sort by FDR ascending, then alphabetically
    plot_df.sort_values(by=['fdr', 'feature'], inplace=True)

    # Reverse to make lowest FDR appear at the top of the horizontal bar chart
    plot_df = plot_df[::-1]

    # Update plotting variables
    labels = plot_df['feature'].tolist()
    odds_ratios = plot_df['odds_ratio'].tolist()
    errors = np.array([plot_df['error_low'].tolist(), plot_df['error_high'].tolist()])
    colors = plot_df['color'].tolist()
    fdr_values = plot_df['fdr'].tolist()

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.6))
    bars = ax.barh(labels, odds_ratios, xerr=errors, color=colors)
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.set_xlabel("Odds Ratio")
    ax.set_title(f"Odds Ratios for {outcome_name} in {name}")

    for i, (bar, fdr) in enumerate(zip(bars, fdr_values)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"FDR={fdr:.3g}", va='center', fontsize=9)

    legend_patches = [
        mpatches.Patch(color=sig_color, label='Significant (FDR < 0.05)'),
        mpatches.Patch(color=non_sig_color, label='Not significant')
    ]
    fig.legend(handles=legend_patches, loc='upper left')
    plt.tight_layout()

    filename = f"{name}_{outcome_name}_horizontal_odds_ratios.png"
    smart_save_fig(path=os.path.join(directory, filename), printer=printer)


def odds_ratio_horizontal(view: Table, name: str, outcomes: Sequence[Outcome], directory: str,
                          printer: Printer = OutPrinter(), max_features: int = MAX_FEATURES_FOR_GRIDS,
                          palette_name: str = DEFAULT_PALETTE_NAME):
    """Skips outcomes that are not binary."""
    if view.n_col() <= max_features:
        view_df = view.to_dataframe()
        for o in outcomes:
            if o.is_binary():
                printer.print(f"Plotting horizontal odds ratio for view {name} and outcome {o.name()}")
                odds_ratio_horizontal_df(
                    view_df=view_df, name=name, outcome_df=o.data(), outcome_name=o.name(),
                    directory=directory, printer=printer, palette_name=palette_name)


def boxplot_first_cols(view: Table, name: str, directory: str, printer: Printer = OutPrinter(), max_features: int = 10):
    n_used_features = min(max_features, view.n_col())
    to_use = view.select_cols(selected=range(n_used_features)).to_dataframe()
    to_use.boxplot()
    smart_save_fig(path=str(os.path.join(directory, name + "_boxplot_cols" + ".png")), printer=printer)


def boxplot_first_rows(view: Table, name, directory, printer: Printer = OutPrinter(), max_box=10):
    to_use = view.transpose().select_cols(selected=range(max_box)).to_dataframe()
    to_use.boxplot()
    smart_save_fig(path=str(os.path.join(directory, name + "_boxplot_rows" + ".png")), printer=printer)


def freedman_diaconis_bin_number(x: SizedIterable, max_bins=100):
    size = len(x)
    if size == 0:
        return 0
    if size > max_bins and size > DEFAULT_MAX_CACHEABLE_CELLS:
        """Very big dataset, computing the result would take time and maybe deplete memory.
        Just use the maximum number of bins."""
        return max_bins
    x = [n for n in x]  # nanpercentile does not really work with just iterables.
    q25 = np.nanpercentile(x, 25)
    q75 = np.nanpercentile(x, 75)
    if math.isnan(q25) or math.isnan(q75):
        return 0
    bin_width = 2*(q75 - q25)*len(x)**(-1/3)
    if bin_width == 0.0:
        return max_bins
    bins = min(round((max(x) - min(x))/bin_width), max_bins)
    return bins


def freedman_diaconis_bin_number_with_tdigest(x: SizedIterable, max_bins=100):
    """Handles very big datasets with tdigest, but this tdigest implementation is very slow."""
    from tdigest import TDigest
    size = len(x)
    digest = TDigest()
    num = 0
    for value in x:
        if not math.isnan(value):
            digest.update(value)
            num += 1
            if num % 10000 == 0:
                print(str(num/size))
    if num == 0:
        return 0
    q25 = digest.percentile(25)
    q75 = digest.percentile(75)
    bin_width = 2*(q75 - q25)*num**(-1/3)
    if bin_width == 0.0:
        return max_bins
    min_value = digest.percentile(0)
    max_value = digest.percentile(100)
    bins = min(round((max_value - min_value)/bin_width), max_bins)
    return bins


def update_histogram(data_chunk: Table, bins, hist):
    """Function to update histogram counts"""
    array_2d=data_chunk.to_numpy()

    # Flatten the 2D array to 1D
    array_1d = array_2d.flatten()

    # Remove NaN values
    clean_array_1d = array_1d[~np.isnan(array_1d)]

    chunk_hist, _ = np.histogram(clean_array_1d, bins=bins)
    hist += chunk_hist
    return hist


def plot_density_histogram_in_chunks(
        data: Table, num_bins: int, directory: str, name: str,
        progress_observer_factory: ProgressObserverFactory = DEFAULT_PROGRESS_OBSERVER_FACTORY):
    n_chunks = data.n_col()
    data_chunks = data.columns()
    min_val = math.inf
    max_val = -math.inf
    po = progress_observer_factory.create_progress_observer(
        job_name="Computing min and max values for density histogram.")
    po.notify_start()
    if not n_chunks >= 1:
        po.notify_message("There are no columns. Nothing to plot. Routine will end.")
        po.notify_end()
        return
    for i, chunk in enumerate(data_chunks):
        chunk_np = chunk.to_numpy()
        min_val = safe_nanmin([min_val, safe_nanmin(chunk_np)])
        max_val = safe_nanmax([max_val, safe_nanmax(chunk_np)])
        po.notify_progress(proportion=i/n_chunks)
    po.notify_end()

    # Define the number of bins
    bins = np.linspace(min_val, max_val, num_bins+1)  # 10 bins from 0 to 100

    # Initialize histogram counts
    hist = np.zeros(num_bins)

    po = progress_observer_factory.create_progress_observer(
        job_name="Updating histogram with each chunk.")
    po.notify_start()
    for i, chunk in enumerate(data_chunks):
        hist = update_histogram(data_chunk=chunk, bins=bins, hist=hist)
        po.notify_progress(proportion=i/n_chunks)
    po.notify_end()

    po = progress_observer_factory.create_progress_observer(
        job_name="Plotting histogram to file.")
    po.notify_start()
    fig = plt.figure()
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black', align='edge')
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(os.path.join(directory, name + "_density.png"), dpi=600, bbox_inches='tight')
    except BaseException as e:
        po.notify_message("Exception while saving figure:\n" + str(e))
    plt.close(fig)
    po.notify_end()


def plot_view(view: Table, name: str, outcomes: Sequence[Outcome],
              directory: str, printer: Printer = OutPrinter(), skip_huge_views: bool = True):
    size = view.size()
    if size > 0:
        odds_ratio_horizontal(view=view, name=name, outcomes=outcomes, directory=directory, printer=printer)
        boxplot_grid(view=view, name=name, outcomes=outcomes, directory=directory, printer=printer)
        printer.print("Plotting box plot of first columns.")
        boxplot_first_cols(view=view, name=name, directory=directory, printer=printer)
        if skip_huge_views and size > DEFAULT_MAX_CACHEABLE_CELLS:
            printer.print("Skipping box plot of first rows for huge view " + str(name))
        else:
            printer.print("Plotting box plot of first rows.")
            boxplot_first_rows(view=view, name=name, directory=directory, printer=printer)
        if skip_huge_views and size > DEFAULT_MAX_CACHEABLE_CELLS:
            printer.print("Skipping density histogram for huge view " + str(name))
        else:
            printer.print("Computing Freedman Diaconis bin number for density histogram.")
            flat = view.flatten()
            num_bins = freedman_diaconis_bin_number(flat)
            po_fact = SmartProgressObserverFactory(printer=printer, minutes_of_quiet=1)
            plot_density_histogram_in_chunks(
                data=view,num_bins=num_bins, directory=directory, name=name, progress_observer_factory=po_fact)
    else:
        printer.print("Cannot plot this view since it contains no data.")


def plot_views(data: InputData, directory, printer: Printer, skip_huge_views: bool = True):
    views = data.views()
    for v in views.keys():
        printer.print("Plotting view " + v)
        plot_view(view=views.view(v), name=v, outcomes=data.outcomes(), directory=directory,
                  printer=printer, skip_huge_views=skip_huge_views)
