from typing import Optional

from input_data.swedish_input_creator import SWEDISH_NICK

GF_LAB = "GF"
RFE_LAB = "RFE"
NSGA2_LAB = "NSGA2"
NSGA3_LAB = "NSGA3"
NSGA2_CH_LAB = "NSGA2-CH"
NSGA2_CHS_LAB = "NSGA2-CHS"
LASSO_MO_LAB = "LASSO-MO"

ALL_GA_LABS = [NSGA2_LAB, NSGA3_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]
ALL_MAIN_WITH_INNER = [RFE_LAB, GF_LAB]
ALL_MAIN_WITH_INNER.extend(ALL_GA_LABS)
ALL_MAIN = [LASSO_MO_LAB]+ALL_MAIN_WITH_INNER
ALL_MAIN_NO_NSGA3 = [LASSO_MO_LAB, RFE_LAB, GF_LAB, NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]

NB_LAB = "NB"
RF_LAB = "RF"
LR_LAB = "LR"
SVM_LAB = "SVM"

ALL_INNER_LABS = [NB_LAB, SVM_LAB, RF_LAB, LR_LAB]

TCGA_BRCA_LAB = "tcga_brca"
TCGA_LU_LAB = "tcga_lu"
TCGA_LU2_LAB = "tcga_lu2"
TCGA_DIG5_LAB = "tcga_dig5"
TCGA_DIG_TYPE_LAB = "tcga_dig_type"
TCGA_KI_LAB = "tcga_ki"
TCGA_KI3_LAB = "tcga_ki3"
TCGA_OV_LAB = "tcga_ov"
TCGA_THCA2_LAB = "tcga_thca2"
TCGA_THCA3_LAB = "tcga_thca3"
TCGA_THCA_BL_LAB = "tcga_thca_bl"
SWEDISH_LAB = SWEDISH_NICK

ALL_CV_DATASETS = [
    TCGA_BRCA_LAB, TCGA_LU_LAB, TCGA_LU2_LAB, TCGA_DIG5_LAB, TCGA_KI_LAB, TCGA_OV_LAB,
    TCGA_THCA3_LAB, TCGA_THCA_BL_LAB, TCGA_DIG_TYPE_LAB]


def main_and_inner_label(main_lab: str, inner_lab: Optional[str]) -> str:
    if inner_lab is None:
        return main_lab
    else:
        return main_lab + " " + inner_lab


def has_inner_model(main_lab: str) -> bool:
    return main_lab != LASSO_MO_LAB
