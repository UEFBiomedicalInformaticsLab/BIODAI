from typing import Optional

from input_data.swedish_input_creator import SWEDISH_NICK

NSGA2_LAB = "NSGA2"
NSGA2_CH_LAB = "NSGA2-CH"
NSGA2_CHS_LAB = "NSGA2-CHS"
LASSO_MO_LAB = "LASSO-MO"

ALL_GA_LABS = [NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]
ALL_MAIN_WITH_INNER = []
ALL_MAIN_WITH_INNER.extend(ALL_GA_LABS)
ALL_MAIN = [LASSO_MO_LAB]+ALL_MAIN_WITH_INNER

NB_LAB = "NB"
RF_LAB = "RF"
LR_LAB = "LR"

ALL_INNER_LABS = [NB_LAB, RF_LAB, LR_LAB]

TCGA_BRCA_LAB = "tcga_brca"
SWEDISH_LAB = SWEDISH_NICK

ALL_CV_DATASETS = [TCGA_BRCA_LAB]


def main_and_inner_label(main_lab: str, inner_lab: Optional[str]) -> str:
    if inner_lab is None:
        return main_lab
    else:
        return main_lab + " " + inner_lab


def has_inner_model(main_lab: str) -> bool:
    return main_lab != LASSO_MO_LAB
