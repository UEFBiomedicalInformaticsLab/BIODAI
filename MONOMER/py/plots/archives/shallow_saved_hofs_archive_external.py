from input_data.input_creators_archive import EXT_DIG_NICK, CPTAC3_SUB_NICK, GSE138042_NO0_NICK, \
    EXT_OV_NICK, CPTAC3_SUB_UQ_NICK, SWEDISH_NEW_NICK, KID_GSE152938_NICK, SWEDISH_NEW2_NICK, CPTAC3_SUB_UQ2_NICK, \
    EXT_OV2_NICK, KID_GSE152938B1_NICK, KID_GSE152938B2_NICK, CPTAC3_SUB_UQ3_NICK, CPTAC3_SUB_UQ4_NICK, \
    KID_GSE152938D_NICK
from input_data.swedish_input_creator import SWEDISH_NICK
from plots.plot_labels import TCGA_BRCA_LAB, SWEDISH_LAB, NB_LAB, RF_LAB, LR_LAB, SVM_LAB, LASSO_MO_LAB, \
    TCGA_DIG_TYPE_LAB, TCGA_LU_LAB, TCGA_THCA2_LAB, TCGA_OV_LAB, TCGA_KI3_LAB, ALL_MAIN, ALL_MAIN_NO_NSGA3, \
    ALL_INNER_LABS
from plots.archives.automated_hofs_archive import default_saved_hof_from_labels_external, \
    all_main_hofs_for_inner_model_external
from plots.saved_external_val import SavedExternalVal

TCGA_BRCA_EXTERNAL_NB = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=NB_LAB)
TCGA_BRCA_EXTERNAL_RF = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=RF_LAB)
TCGA_BRCA_EXTERNAL_LR = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=LR_LAB)
TCGA_BRCA_EXTERNAL_SVM = all_main_hofs_for_inner_model_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, inner_lab=SVM_LAB)

TCGA_BRCA_EXTERNAL_LASSO = default_saved_hof_from_labels_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB, main_lab=LASSO_MO_LAB)


def all_external_validations(main_labs: list[str] = ALL_MAIN) -> list[SavedExternalVal]:
    return [
        SavedExternalVal(internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NEW_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NEW2_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ2_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ3_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ4_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_DIG_TYPE_LAB, external_nick=EXT_DIG_NICK, main_labs=main_labs),
        # SavedExternalVal(internal_label=TCGA_THCA2_LAB, external_nick=GSE138042_NICK),
        SavedExternalVal(internal_label=TCGA_THCA2_LAB, external_nick=GSE138042_NO0_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938B1_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938B2_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938D_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_OV_LAB, external_nick=EXT_OV_NICK, main_labs=main_labs),
        SavedExternalVal(internal_label=TCGA_OV_LAB, external_nick=EXT_OV2_NICK, main_labs=main_labs)
    ]


def cattelani2023_external_validations(main_labs: list[str] = ALL_MAIN_NO_NSGA3,
                                       inner_labs: list[str] = ALL_INNER_LABS) -> list[SavedExternalVal]:
    return [
        SavedExternalVal(
            internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NICK, main_labs=main_labs, inner_labs=inner_labs),
        SavedExternalVal(
            internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938D_NICK, main_labs=main_labs, inner_labs=inner_labs),
        SavedExternalVal(
            internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ4_NICK, main_labs=main_labs, inner_labs=inner_labs),
        SavedExternalVal(
            internal_label=TCGA_OV_LAB, external_nick=EXT_OV2_NICK, main_labs=main_labs, inner_labs=inner_labs)
    ]


ALL_EXTERNAL_VALIDATIONS = all_external_validations()
