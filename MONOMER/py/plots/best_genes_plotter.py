from collections.abc import Sequence

from pandas import DataFrame

from input_data.view_prefix import remove_view_prefix
from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_cv, flatten_hofs_for_dataset_external
from plots.archives.shallow_saved_hofs_archive_external import all_external_validations
from plots.plot_labels import ALL_CV_DATASETS, ALL_MAIN_NO_NSGA3
from plots.solution_utils import solutions_and_hof_names
from plots.saved_hof import SavedHoF
from plots.summary_statistics_plotter import SUMMARY_STAT_DIR
from plots.table_plot import plot_table_to_file
from saved_solutions.saved_solution import union_of_features, average_individual
from util.dataframes import n_row
import matplotlib.colors as mc
import colorsys

BEST_GENES_STR = "best_genes"
N_BEST_GENES = 6
MAIN_LABS = ALL_MAIN_NO_NSGA3


def adjust_lightness(color, amount=1.0):
    """Above 1 gets lighter, below one gets darker. With 2 gets white."""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0.0, min(1.0, amount * c[1])), c[2])


def process_dataset(save_path: str, hofs: Sequence[SavedHoF]):
    solutions, hof_names = solutions_and_hof_names(hofs=hofs)
    colnames = ["Algorithm"] + [str(i) for i in range(1, N_BEST_GENES + 1)]
    table_to_plot = DataFrame(columns=colnames)
    inner_cells_colour = []
    for alg_name, alg_solutions in zip(hof_names, solutions):
        all_features = union_of_features(alg_solutions)
        average_ind = average_individual(alg_solutions, all_features)
        freq_df = DataFrame(data={'feature': all_features, 'frequency': average_ind})
        freq_df = freq_df.sort_values(by=['frequency'], ascending=False)
        freq_df = freq_df.iloc[0:N_BEST_GENES, ]
        for i in range(max(0, N_BEST_GENES - n_row(freq_df))):
            freq_df = freq_df.append({'feature': "", 'frequency': 0}, ignore_index=True)
        for i in range(N_BEST_GENES):
            if freq_df.iloc[i, 0] != "":
                freq_df.iloc[i, 0] = remove_view_prefix(freq_df.iloc[i, 0])[0]
        table_to_plot.loc[len(table_to_plot)] = [alg_name] + list(freq_df.iloc[:, 0])
        frequencies = freq_df.iloc[:, 1]
        inner_cells_colour.append(['w'] + [adjust_lightness(color="cornflowerblue", amount=2-f) for f in frequencies])
    plot_table_to_file(path=save_path, df=table_to_plot, inner_cells_colour=inner_cells_colour)


if __name__ == '__main__':
    for dataset_label in ALL_CV_DATASETS:
        plot_path = SUMMARY_STAT_DIR + "/cv/" + dataset_label + "/" + BEST_GENES_STR
        hofs = flatten_hofs_for_dataset_cv(dataset_lab=dataset_label, main_labs=MAIN_LABS)
        print("Processing dataset " + str(dataset_label))
        process_dataset(save_path=plot_path, hofs=hofs)
    for ext in all_external_validations(main_labs=MAIN_LABS):
        external_hofs = ext.nested_hofs()
        internal_label = ext.internal_label()
        external_nick = ext.external_nick()
        print("Processing external validation " + str(ext.internal_label() + " - " + external_nick))
        hofs = flatten_hofs_for_dataset_external(
            dataset_lab=internal_label, external_nick=external_nick, main_labs=MAIN_LABS)
        plot_path = SUMMARY_STAT_DIR + "/external/" + internal_label + "_" + external_nick + "/" + BEST_GENES_STR
        process_dataset(save_path=plot_path, hofs=hofs)
