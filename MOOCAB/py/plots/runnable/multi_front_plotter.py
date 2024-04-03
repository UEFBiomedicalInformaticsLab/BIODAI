from plots.monotonic_front import multi_front_plot_from_saved_hofs
from plots.archives.shallow_saved_hofs_archive_cv import TCGA_BRCA_1000_NB, TCGA_BRCA_1000_RF, TCGA_BRCA_1000_LOGIT, \
    TCGA_BRCA_1000, \
    TCGA_BRCA_LASSO, TCGA_BRCA_1000_FULL_OPTIONAL, TCGA_PRAD_1000_MRNA_CHS, TCGA_PRAD_1000_MRNA_LASSO, \
    TCGA_BRCA_RFE, TCGA_BRCA_GF, TCGA_BRCA_1000_FO_NB

X_MIN = None  # -1
X_MAX = None  # 46
Y_MIN = 0.18
Y_MAX = 0.9


if __name__ == '__main__':
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_1000_NB, save_path="comparison1000/NB",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_1000_RF, save_path="comparison1000/RF",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_1000_LOGIT, save_path="comparison1000/logit",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_1000, save_path="comparison1000",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=[TCGA_BRCA_LASSO], save_path="comparison1000/lasso",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_1000_FULL_OPTIONAL, save_path="comparison1000/full_optional",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=27, y_min=Y_MIN, y_max=Y_MAX)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_PRAD_1000_MRNA_CHS, save_path="comparison1000/prad_mrna/chs",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=None, y_min=None, y_max=None)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_PRAD_1000_MRNA_LASSO, save_path="comparison1000/prad_mrna/lasso",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=None, y_min=None, y_max=None)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_RFE,
                                     save_path="comparison1000/tcga_brca/rfe",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=None, y_min=None, y_max=None)
    multi_front_plot_from_saved_hofs(saved_hofs=TCGA_BRCA_RFE + TCGA_BRCA_GF,
                                     save_path="comparison1000/tcga_brca/rfe_gf",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=None, y_min=None, y_max=None)
    multi_front_plot_from_saved_hofs(saved_hofs=[TCGA_BRCA_LASSO] + [TCGA_BRCA_1000_FO_NB] + TCGA_BRCA_RFE + TCGA_BRCA_GF,
                                     save_path="comparison1000/tcga_brca/rfe_gf_lasso",
                                     x_label="number of features", y_label="balanced accuracy",
                                     x_min=None, x_max=None, y_min=None, y_max=None)
