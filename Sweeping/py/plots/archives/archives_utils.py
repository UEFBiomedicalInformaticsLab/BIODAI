from collections.abc import Sequence
from typing import Optional, Union

from input_data.input_creator.input_creators_archive import KIRC_MV_NICK, SARC_MV_NICK, LGG_MV_NICK
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from plots.plot_labels import NB_LAB, RF_LAB, LR_LAB, SVM_LAB, TREE_LAB, KIRC_MV_LAB, SARC_MV_LAB, LGG_MV_LAB
from univariate_feature_selection.univariate_feature_selector_descriptor import ANOVA_NICK, LOGISTIC_FDR_SELECTOR_NICK

DEFAULT_COX_FI = True


DATASET_BASE_DIR_MAP = {
    KIRC_MV_LAB: KIRC_MV_NICK,
    SARC_MV_LAB: SARC_MV_NICK,
    LGG_MV_LAB: LGG_MV_NICK
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


def categorical_fs_uses_fdr(categorical_fs_nick: str) -> bool:
    if categorical_fs_nick == ANOVA_NICK:
        return False
    if categorical_fs_nick == LOGISTIC_FDR_SELECTOR_NICK:
        return True
    raise ValueError("Unexpected categorical feature selector.")


def categorical_fs_uses_covariates(categorical_fs_nick: str) -> bool:
    if categorical_fs_nick == ANOVA_NICK:
        return False
    if categorical_fs_nick == LOGISTIC_FDR_SELECTOR_NICK:
        return True
    raise ValueError("Unexpected categorical feature selector.")


def fdr_thresholds_to_use(categorical_fs_nick: str, fdr_thresholds: Sequence[float]) -> Sequence[Union[float, None]]:
    if categorical_fs_uses_fdr(categorical_fs_nick=categorical_fs_nick):
        return fdr_thresholds
    else:
        return [None]
