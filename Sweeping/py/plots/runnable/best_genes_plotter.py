from collections.abc import Sequence

import pandas as pd
from pandas import DataFrame

from input_data.view_prefix import remove_view_prefix
from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_external, all_hof_combinations_cv
from plots.archives.shallow_saved_hofs_archive_external import all_external_validations
from plots.default_labels_map import LabelsTransformer, DEFAULT_LABELS_TRANSFORMER
from plots.plot_labels import ALL_CV_DATASETS, ALL_MAIN_NO_NSGA3
from plots.solution_utils import solutions_and_hof_names
from plots.saved_hof import SavedHoF
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR
from plots.table_plot import plot_table_to_file
from saved_solutions.saved_solution import union_of_features, average_individual
from util.table.table_utils import n_row
import matplotlib.colors as mc
import colorsys

BEST_FEATURES_STR = "best_features"
N_BEST_GENES_FOR_PLOT = 6
N_BEST_GENES_FOR_CSV = 20
MAIN_LABS = ALL_MAIN_NO_NSGA3


def adjust_lightness(color, amount=1.0):
    """Above 1 gets lighter, below one gets darker. With 2 gets white."""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0.0, min(1.0, amount * c[1])), c[2])


def best_features_table(
        hofs: Sequence[SavedHoF], labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER,
        show_counts: bool = False, n_best_genes: int = N_BEST_GENES_FOR_CSV) -> tuple[DataFrame, DataFrame]:
    solutions, hof_names = solutions_and_hof_names(hofs=hofs)
    colnames = ["Algorithm"] + [str(i) for i in range(1, n_best_genes + 1)]
    table_to_plot = DataFrame(columns=colnames)
    frequencies_df = DataFrame(columns=colnames)
    for alg_name, alg_solutions in zip(hof_names, solutions):
        n_alg_solutions = len(alg_solutions)
        if n_alg_solutions > 0:
            all_features = union_of_features(alg_solutions)
            average_ind = average_individual(alg_solutions, all_features)
            freq_df = DataFrame(data={'feature': all_features, 'frequency': average_ind})
            for i in range(len(freq_df)):
                if freq_df.iloc[i, 0] != "":
                    # .iat is a tiny bit faster than .iloc for scalar set
                    freq_df.iat[i, 0] = remove_view_prefix(freq_df.iat[i, 0])[0]
            # Sort first by frequency and then by feature name.
            freq_df = freq_df.sort_values(by=['frequency', 'feature'], ascending=[False, True], kind='mergesort')
            freq_df = freq_df.iloc[0:n_best_genes,]
            for i in range(max(0, n_best_genes - n_row(freq_df))):
                freq_df = pd.concat(
                    [freq_df, pd.DataFrame({'feature': ("",), 'frequency': (0,)})], ignore_index=True)
            for i in range(n_best_genes):
                if freq_df.iloc[i, 0] != "":
                    if show_counts:
                        freq_df.iloc[i, 0] = freq_df.iloc[i, 0] + " " + str(round(freq_df.iloc[i, 1] * n_alg_solutions))
            transformed_alg_name = labels_transformer.apply(alg_name)
            table_to_plot.loc[len(table_to_plot)] = [transformed_alg_name] + list(freq_df.iloc[:, 0])
            frequencies_df.loc[len(table_to_plot)] = [transformed_alg_name] + list(freq_df.iloc[:, 1])
    return table_to_plot, frequencies_df


def best_features_plotter_process_dataset(
        save_path: str, hofs: Sequence[SavedHoF], labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER,
        show_counts: bool = False):
    """Save path should be a file name without extension, the different files will be saved starting from that path."""
    table_to_plot, frequencies_df = best_features_table(
        hofs=hofs, labels_transformer=labels_transformer,
        show_counts=show_counts)
    inner_cells_colour = []
    for i in range(len(frequencies_df)):
        inner_cells_colour.append(
            ['w'] + [adjust_lightness(color="cornflowerblue", amount=2 - f)
                     for f in frequencies_df.iloc[i, 1:N_BEST_GENES_FOR_PLOT+1]])
    plot_table_to_file(path=save_path, df=table_to_plot.iloc[:, 0:N_BEST_GENES_FOR_PLOT+1],
                       inner_cells_colour=inner_cells_colour)
    table_to_plot.to_csv(save_path + ".csv", index=False)
    frequencies_df.to_csv(save_path + "_frequencies.csv", index=False)


if __name__ == '__main__':
    for dataset_label in ALL_CV_DATASETS:
        plot_path = SUMMARY_STAT_DIR + "/cv/" + dataset_label + "/" + BEST_FEATURES_STR
        hofs = all_hof_combinations_cv(dataset_lab=dataset_label, main_labs=MAIN_LABS)
        print("Processing dataset " + str(dataset_label))
        best_features_plotter_process_dataset(save_path=plot_path, hofs=hofs)
    for ext in all_external_validations(main_labs=MAIN_LABS):
        external_hofs = ext.nested_hofs()
        internal_label = ext.internal_label()
        external_nick = ext.external_nick()
        print("Processing external validation " + str(ext.internal_label() + " - " + external_nick))
        hofs = flatten_hofs_for_dataset_external(
            dataset_lab=internal_label, external_nick=external_nick, main_labs=MAIN_LABS)
        plot_path = SUMMARY_STAT_DIR + "/external/" + internal_label + "_" + external_nick + "/" + BEST_FEATURES_STR
        best_features_plotter_process_dataset(save_path=plot_path, hofs=hofs)
