from input_data.input_creator.input_creators_archive import EXT_DIG_NICK, CPTAC3_SUB_NICK, GSE138042_NO0_NICK, \
    EXT_OV_NICK, CPTAC3_SUB_UQ_NICK, SWEDISH_NEW_NICK, KID_GSE152938_NICK, SWEDISH_NEW2_NICK, CPTAC3_SUB_UQ2_NICK, \
    EXT_OV2_NICK, KID_GSE152938B1_NICK, KID_GSE152938B2_NICK, CPTAC3_SUB_UQ3_NICK, CPTAC3_SUB_UQ4_NICK, \
    KID_GSE152938D_NICK, KID_GSE152938D_T2_NICK
from input_data.input_creator.swedish_input_creator import SWEDISH_NICK
from plots.plot_labels import (TCGA_BRCA_LAB, TCGA_DIG_TYPE_LAB, TCGA_LU_LAB, TCGA_THCA2_LAB, TCGA_OV_LAB, TCGA_KI3_LAB,
                               ALL_MAIN_LABS, TCGA_KI_LAB)
from plots.saved_external_val import SavedExternalVal


def all_external_validations(main_labs: list[str] = ALL_MAIN_LABS) -> list[SavedExternalVal]:
    return [
        SavedExternalVal(internal_label=TCGA_KI_LAB, external_nick=KID_GSE152938D_T2_NICK, main_labs=main_labs),
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


ALL_EXTERNAL_VALIDATIONS = all_external_validations()
