from input_data.swedish_input_creator import SWEDISH_NICK
from plots.plot_labels import TCGA_BRCA_LAB, SWEDISH_LAB, NB_LAB, RF_LAB, LR_LAB, LASSO_MO_LAB
from plots.archives.automated_hofs_archive import default_saved_hof_from_labels_external, \
    all_main_hofs_for_inner_model_external
from plots.saved_external_val import SavedExternalVal

TCGA_BRCA_EXTERNAL_NB = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=NB_LAB)
TCGA_BRCA_EXTERNAL_RF = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=RF_LAB)
TCGA_BRCA_EXTERNAL_LR = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=LR_LAB)

TCGA_BRCA_EXTERNAL_LASSO = default_saved_hof_from_labels_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, main_lab=LASSO_MO_LAB)

ALL_EXTERNAL_VALIDATIONS = [SavedExternalVal(internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NICK)]
