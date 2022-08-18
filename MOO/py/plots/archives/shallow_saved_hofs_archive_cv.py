from plots.plot_labels import TCGA_BRCA_LAB, NB_LAB, NSGA2_CHS_LAB, RF_LAB, LR_LAB, LASSO_MO_LAB
from plots.saved_hof import SavedHoF
from plots.archives.automated_hofs_archive import default_saved_hof_from_labels_cv, all_ga_hofs_with_inner_model_cv

TCGA_BRCA_1000_FO_NB = default_saved_hof_from_labels_cv(dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_CHS_LAB, inner_lab=NB_LAB)
TCGA_BRCA_1000_FO_RF = default_saved_hof_from_labels_cv(dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_CHS_LAB, inner_lab=RF_LAB)
TCGA_BRCA_1000_FO_LOGIT = default_saved_hof_from_labels_cv(dataset_lab=TCGA_BRCA_LAB, main_lab=NSGA2_CHS_LAB, inner_lab=LR_LAB)

TCGA_BRCA_1000_NB = all_ga_hofs_with_inner_model_cv(dataset_lab=TCGA_BRCA_LAB, inner_lab=NB_LAB)
TCGA_BRCA_1000_RF = all_ga_hofs_with_inner_model_cv(dataset_lab=TCGA_BRCA_LAB, inner_lab=RF_LAB)
TCGA_BRCA_1000_LOGIT = all_ga_hofs_with_inner_model_cv(dataset_lab=TCGA_BRCA_LAB, inner_lab=LR_LAB)

TCGA_BRCA_LASSO = default_saved_hof_from_labels_cv(dataset_lab=TCGA_BRCA_LAB, main_lab=LASSO_MO_LAB)

NB_SYMMETRIC_1000 = SavedHoF(
    path="brca/mrna/NB_bal_acc_leanness/5_folds/NSGA2_k3_pop500_uni0-50_gen1000_CrowdCI_c0.33_m1.0symm_(MV_lassoFI,none)/hofs/Pareto",
    name="NSGA2-CH symm NB")

TCGA_BRCA_1000_FULL_OPTIONAL = [
    TCGA_BRCA_LASSO, TCGA_BRCA_1000_FO_RF, TCGA_BRCA_1000_FO_LOGIT, TCGA_BRCA_1000_FO_NB
]

TCGA_BRCA_1000 = []
TCGA_BRCA_1000.extend(TCGA_BRCA_1000_RF)
TCGA_BRCA_1000.extend(TCGA_BRCA_1000_LOGIT)
TCGA_BRCA_1000.extend(TCGA_BRCA_1000_NB)
TCGA_BRCA_1000.append(TCGA_BRCA_LASSO)
