from typing import Optional

CH_SUFF = "-CH"
CHS_SUFF = "-CHS"
CHP_SUFF = "-CHP"

NSGA_LAB = "NSGA"
NSGA2_LAB = NSGA_LAB+"2"
NSGA3_LAB = NSGA_LAB+"3"
NSGA2_CH_LAB = NSGA2_LAB+CH_SUFF
NSGA2_CHS_LAB = NSGA2_LAB+CHS_SUFF
NSGA2_CHP_LAB = NSGA2_LAB+CHP_SUFF
NSGA3_CH_LAB = NSGA3_LAB+CH_SUFF
NSGA3_CHS_LAB = NSGA3_LAB+CHS_SUFF
NSGA3_CHP_LAB = NSGA3_LAB+CHP_SUFF
ALL_NSGA_LABS = [NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB, NSGA2_CHP_LAB,
                 NSGA3_LAB, NSGA3_CH_LAB, NSGA3_CHS_LAB, NSGA3_CHP_LAB]

ALL_GA_LABS = ALL_NSGA_LABS

ALL_MAIN_WITH_INNER = ALL_GA_LABS
ALL_MAIN_LABS = ALL_MAIN_WITH_INNER
SELECTED_MAIN_LABS = [NSGA3_CHS_LAB, NSGA3_CHP_LAB]
ALL_MAIN_NO_NSGA3 = [NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]

NB_LAB = "NB"
RF_LAB = "RF"
RF_LEGACY_LAB = "RF_old"
LR_LAB = "LR"
SVM_LAB = "SVM"
TREE_LAB = "tree"

ALL_INNER_LABS = [NB_LAB, SVM_LAB, TREE_LAB, RF_LAB, LR_LAB, RF_LEGACY_LAB]
SELECTED_INNER_LABS = [NB_LAB, SVM_LAB]

TCGA_KID_IHC_DET_LAB = "tcga_ki_ihc_det"
TCGA_KID_IHC_OS_LAB = "tcga_ki_ihc_os"

ALL_CV_DATASETS = [TCGA_KID_IHC_DET_LAB, TCGA_KID_IHC_OS_LAB]


def main_and_inner_label(main_lab: str, inner_lab: Optional[str], adjuster_regressor: Optional[str] = None) -> str:
    """For example: NSGA2 NB."""
    res = ""
    if adjuster_regressor is not None:
        res += adjuster_regressor + " "
    res += main_lab
    if inner_lab is not None:
        res += " " + inner_lab
    return res


def has_classification_inner_model(main_lab: str) -> bool:
    return True
