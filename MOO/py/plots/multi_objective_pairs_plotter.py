from plots.default_labels_map import default_labels_map
from plots.objective_pairs_plots import external_objective_pairs_plot_from_saved_hofs
from plots.archives.shallow_saved_hofs_archive_external import TCGA_BRCA_EXTERNAL_NB, TCGA_BRCA_EXTERNAL_RF, \
    TCGA_BRCA_EXTERNAL_LR, TCGA_BRCA_EXTERNAL_LASSO

X_MIN = -2
X_MAX = 53
Y_MIN = None
Y_MAX = 0.83


if __name__ == '__main__':
    external_objective_pairs_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_EXTERNAL_NB,
                                                  save_path="objective_pairs_comparison_NB",
                                                  x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                                                  labels_map=default_labels_map)
    external_objective_pairs_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_EXTERNAL_RF,
                                                  save_path="objective_pairs_comparison_RF",
                                                  x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                                                  labels_map=default_labels_map)
    external_objective_pairs_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_EXTERNAL_LR,
                                                  save_path="objective_pairs_comparison_logit",
                                                  x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                                                  labels_map=default_labels_map)
    external_objective_pairs_plot_from_saved_hofs(saved_hofs=[TCGA_BRCA_EXTERNAL_LASSO],
                                                  save_path="objective_pairs_comparison_lasso",
                                                  x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                                                  labels_map=default_labels_map)
