from string import ascii_lowercase

from matplotlib import pyplot as plt

from plots.archives.automated_hofs_archive import default_saved_hof_from_labels_cv
from plots.archives.shallow_saved_hofs_archive_cv import TCGA_BRCA_LASSO
from plots.folds_scatter_plots import folds_scatter_plots_from_saved_hof_one_alg_to_ax
from plots.plot_labels import TCGA_BRCA_LAB, NSGA2_LAB, NB_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB
from plots.plot_utils import smart_save_fig
from util.grid import row_col_by_index
from util.utils import ceil_division

if __name__ == '__main__':
    n_subplots = 4
    ncols = 2
    x_label = "number of features"
    y_label = "balanced accuracy"
    save_path = "folds_scatter_plots/comparison1000/eccb_fig1.png"
    y_min = 0.180
    y_max = 1.020
    x_min = -1
    ga_x_max = 59
    alpha = 0.75
    leanness_str = "leanness"

    nrows = ceil_division(num=n_subplots, den=ncols)
    figsize_x = 5.0*ncols + 1.0
    figsize_y = 4.0*nrows + 1.0
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=600, figsize=(figsize_x, figsize_y), sharex=False, sharey=True)
    n_boxes = nrows * ncols

    i = 0
    row, col = row_col_by_index(index=i, ncol=ncols)
    plot_axs = axs[row, col]
    if i < n_subplots:
        letter = ascii_lowercase[i]
        plot_axs.set_title(letter)
        folds_scatter_plots_from_saved_hof_one_alg_to_ax(
            saved_hof=TCGA_BRCA_LASSO,
            ax=plot_axs, x_objective=leanness_str, y_objective="bal_acc",
            x_min=x_min, x_max=None, y_min=y_min, y_max=y_max, alpha=alpha,
            x_label=x_label, y_label=y_label)
        plot_axs.set(xlabel=None)
        plot_axs.set(ylabel=None)

    i = 1
    row, col = row_col_by_index(index=i, ncol=ncols)
    plot_axs = axs[row, col]
    if i < n_subplots:
        letter = ascii_lowercase[i]
        plot_axs.set_title(letter)
        folds_scatter_plots_from_saved_hof_one_alg_to_ax(
            saved_hof=default_saved_hof_from_labels_cv(
                dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_LAB, inner_lab=NB_LAB),
            ax=plot_axs, x_objective=leanness_str, y_objective="NB_bal_acc",
            x_min=x_min, x_max=ga_x_max, y_min=y_min, y_max=y_max, alpha=alpha,
            x_label=x_label, y_label=y_label)
        plot_axs.set(xlabel=None)
        plot_axs.set(ylabel=None)

    i = 2
    row, col = row_col_by_index(index=i, ncol=ncols)
    plot_axs = axs[row, col]
    if i < n_subplots:
        letter = ascii_lowercase[i]
        plot_axs.set_title(letter)
        folds_scatter_plots_from_saved_hof_one_alg_to_ax(
            saved_hof=default_saved_hof_from_labels_cv(
                dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_CH_LAB, inner_lab=NB_LAB),
            ax=plot_axs, x_objective=leanness_str, y_objective="NB_bal_acc",
            x_min=x_min, x_max=ga_x_max, y_min=y_min, y_max=y_max, alpha=alpha,
            x_label=x_label, y_label=y_label)
        plot_axs.set(xlabel=None)
        plot_axs.set(ylabel=None)

    i = 3
    row, col = row_col_by_index(index=i, ncol=ncols)
    plot_axs = axs[row, col]
    if i < n_subplots:
        letter = ascii_lowercase[i]
        plot_axs.set_title(letter)
        folds_scatter_plots_from_saved_hof_one_alg_to_ax(
            saved_hof=default_saved_hof_from_labels_cv(
                dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_CHS_LAB, inner_lab=NB_LAB),
            ax=plot_axs, x_objective=leanness_str, y_objective="NB_bal_acc",
            x_min=x_min, x_max=ga_x_max, y_min=y_min, y_max=y_max, alpha=alpha,
            x_label=x_label, y_label=y_label)
        plot_axs.set(xlabel=None)
        plot_axs.set(ylabel=None)

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.tight_layout()
    smart_save_fig(path=save_path)
