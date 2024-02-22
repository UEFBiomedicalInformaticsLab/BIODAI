from typing import Optional

from input_data.brca_input_creator import BrcaInputCreator
from input_data.input_creators_archive import TCGA_DIG5_NICK, TCGA_DIG_TYPE_NICK, \
    TCGA_KIR3_NICK, TCGA_KIDNEY_IHC_NICK, \
    TCGA_KIDNEY_IHC_DET_NICK, TCGA_KIDNEY_IHC_DET_OS_NICK
from input_data.kir_input_creator import KirInputCreator
from input_data.swedish_input_creator import SWEDISH_NICK
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from plots.plot_labels import TCGA_BRCA_LAB, TCGA_DIG5_LAB, TCGA_DIG_TYPE_LAB, TCGA_KI_LAB, \
    TCGA_KI3_LAB, NB_LAB, RF_LAB, \
    LR_LAB, SVM_LAB, TREE_LAB, TCGA_KID_IHC_LAB, TCGA_KID_IHC_DET_LAB, TCGA_KID_IHC_OS_LAB, \
    SWEDISH_LAB

DATASET_BASE_DIR_MAP = {
    TCGA_BRCA_LAB: BrcaInputCreator().nick(),
    TCGA_DIG5_LAB: TCGA_DIG5_NICK,
    TCGA_DIG_TYPE_LAB: TCGA_DIG_TYPE_NICK,
    TCGA_KI_LAB: KirInputCreator().nick(),
    TCGA_KI3_LAB: TCGA_KIR3_NICK,
    TCGA_KID_IHC_LAB: TCGA_KIDNEY_IHC_NICK,
    TCGA_KID_IHC_DET_LAB: TCGA_KIDNEY_IHC_DET_NICK,
    TCGA_KID_IHC_OS_LAB: TCGA_KIDNEY_IHC_DET_OS_NICK,
    SWEDISH_LAB: SWEDISH_NICK,
}


def dataset_base_dir(dataset_lab: str) -> str:
    if dataset_lab in DATASET_BASE_DIR_MAP:
        return DATASET_BASE_DIR_MAP[dataset_lab]
    else:
        raise ValueError("Unknown dataset label: " + str(dataset_lab))


def inner_lab_to_nick(inner_lab: Optional[str]) -> str:
    if inner_lab is None:
        return ""
    elif inner_lab == NB_LAB:
        return "NB"
    elif inner_lab == RF_LAB:
        return "RF"
    elif inner_lab == LR_LAB:
        return "logit100"
    elif inner_lab == SVM_LAB:
        return "svm"
    elif inner_lab == TREE_LAB:
        return "tree"
    else:
        raise ValueError("Unknown inner model label: " + str(inner_lab))


def inner_lab_to_bal_acc_nick(inner_lab: Optional[str]) -> str:
    bal_acc_nick = BalancedAccuracy().nick()
    inner_model_nick = inner_lab_to_nick(inner_lab=inner_lab)
    if inner_model_nick == "":
        return bal_acc_nick
    else:
        return inner_model_nick + "_" + bal_acc_nick
