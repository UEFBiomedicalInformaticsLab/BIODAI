from plots.folds_scatter_plots import folds_scatter_plots_from_saved_hofs
from plots.runnable.multi_front_plotter import TCGA_BRCA_1000_LOGIT, TCGA_BRCA_1000_NB, TCGA_BRCA_1000_RF,\
    TCGA_BRCA_LASSO

X_MIN = -1
X_MAX = 59
Y_MIN = 0.180
Y_MAX = 1.020


folds_scatter_plots_from_saved_hofs(
    saved_hofs=TCGA_BRCA_1000_LOGIT, save_path="folds_scatter_plots/comparison1000",
    x_objective="leanness", y_objective="logit100_bal_acc",
    x_min=X_MIN, y_min=Y_MIN,
    x_max=X_MAX, y_max=Y_MAX,
    x_label="number of features", y_label="balanced accuracy")

folds_scatter_plots_from_saved_hofs(
    saved_hofs=TCGA_BRCA_1000_NB, save_path="folds_scatter_plots/comparison1000",
    x_objective="leanness", y_objective="NB_bal_acc",
    x_min=X_MIN, y_min=Y_MIN,
    x_max=X_MAX, y_max=Y_MAX,
    x_label="number of features", y_label="balanced accuracy")

folds_scatter_plots_from_saved_hofs(
    saved_hofs=TCGA_BRCA_1000_RF, save_path="folds_scatter_plots/comparison1000",
    x_objective="leanness", y_objective="RF_bal_acc",
    x_min=X_MIN, y_min=Y_MIN,
    x_max=X_MAX, y_max=Y_MAX,
    x_label="number of features", y_label="balanced accuracy")

folds_scatter_plots_from_saved_hofs(
    saved_hofs=[TCGA_BRCA_LASSO], save_path="folds_scatter_plots/comparison1000",
    x_objective="leanness", y_objective="bal_acc",
    x_min=X_MIN, y_min=Y_MIN,
    x_max=X_MAX, y_max=Y_MAX,
    x_label="number of features", y_label="balanced accuracy")
