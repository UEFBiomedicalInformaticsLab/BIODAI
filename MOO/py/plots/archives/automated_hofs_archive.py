from typing import Optional, Sequence

from input_data.brca_input_creator import BrcaInputCreator
from plots.plot_labels import TCGA_BRCA_LAB, NB_LAB, RF_LAB, LR_LAB, NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB, \
    LASSO_MO_LAB, main_and_inner_label, ALL_GA_LABS, ALL_INNER_LABS, \
    SWEDISH_LAB, ALL_CV_DATASETS, ALL_MAIN, has_inner_model
from plots.saved_hof import SavedHoF


def dataset_base_dir(dataset_lab: str) -> str:
    if dataset_lab == TCGA_BRCA_LAB:
        return BrcaInputCreator().nick()
    else:
        raise ValueError("Unknown dataset label: " + str(dataset_lab))


def objectives_dir_from_label(inner_lab: Optional[str]) -> str:
    if inner_lab is None:
        return "bal_acc_leanness"
    elif inner_lab == NB_LAB:
        return "NB_bal_acc_leanness"
    elif inner_lab == RF_LAB:
        return "RF_bal_acc_leanness"
    elif inner_lab == LR_LAB:
        return "leanness_logit100_bal_acc"
    else:
        raise ValueError("Unknown inner model label: " + str(inner_lab))


def optimizer_dir_from_labels(main_lab: str, inner_lab: Optional[str]) -> str:
    pop_size = "500"
    if inner_lab == LR_LAB:
        pop_size = "200"
    if main_lab == NSGA2_LAB:
        return "NSGA2_k3_pop" + pop_size + "_uni0-50_gen1000_CrowdFull_c0.33_m1.0flip_(none,none)"
    elif main_lab == NSGA2_CH_LAB:
        return "NSGA2_k3_pop" + pop_size + "_uni0-50_gen1000_CrowdCI_c0.33_m1.0flip_(MV_lassoFI,none)"
    elif main_lab == NSGA2_CHS_LAB:
        return "NSGA2_k3_pop" + pop_size + "_uni0-50_gen1000_CrowdCICR_c0.33_m1.0symm_(MV_lassoFI,none)"
    elif main_lab == LASSO_MO_LAB:
        return "LASSO_MO"
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))


def hof_dir_from_label(main_lab: str) -> str:
    if main_lab == LASSO_MO_LAB:
        return "LASSO"
    else:
        return "Pareto"


def default_saved_hof_from_labels_cv(dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None) -> SavedHoF:
    if not has_inner_model(main_lab=main_lab):
        inner_lab = None
    name = main_and_inner_label(main_lab=main_lab, inner_lab=inner_lab)
    path = ""
    path += dataset_base_dir(dataset_lab=dataset_lab)
    path += "/mrna/"
    path += objectives_dir_from_label(inner_lab=inner_lab)
    path += "/5_folds/"
    path += optimizer_dir_from_labels(main_lab=main_lab, inner_lab=inner_lab)
    path += "/hofs/"
    path += hof_dir_from_label(main_lab=main_lab)
    return SavedHoF(name=name, path=path)


def all_ga_hofs_with_inner_model_cv(dataset_lab: str, inner_lab: Optional[str]) -> Sequence[SavedHoF]:
    return [default_saved_hof_from_labels_cv(dataset_lab=dataset_lab, main_lab=ga, inner_lab=inner_lab) for ga in ALL_GA_LABS]


def all_inner_hofs_for_main_cv(dataset_lab: str, main_lab: str) -> Sequence[SavedHoF]:
    return [default_saved_hof_from_labels_cv(dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inn) for inn in ALL_INNER_LABS]


def default_saved_hof_from_labels_external(
        dataset_lab: str, external_nick: str, main_lab: str, inner_lab: Optional[str] = None) -> SavedHoF:
    name = main_and_inner_label(main_lab=main_lab, inner_lab=inner_lab)
    path = ""
    path += dataset_base_dir(dataset_lab=dataset_lab)
    path += "/mrna/"
    path += objectives_dir_from_label(inner_lab=inner_lab)
    path += "/external_validation/"
    path += external_nick
    path += "/"
    path += optimizer_dir_from_labels(main_lab=main_lab, inner_lab=inner_lab)
    path += "/hofs/"
    path += hof_dir_from_label(main_lab=main_lab)
    return SavedHoF(name=name, path=path)


def all_hof_combinations_cv(
        dataset_lab: str, main_labs: Sequence[str], inner_labs: Sequence[str]) -> Sequence[SavedHoF]:
    res = []
    for m in main_labs:
        if not has_inner_model(m):
            res.append(default_saved_hof_from_labels_cv(dataset_lab=dataset_lab, main_lab=m, inner_lab=None))
    for inn in inner_labs:
        for m in main_labs:
            if has_inner_model(m):
                res.append(default_saved_hof_from_labels_cv(
                    dataset_lab=dataset_lab, main_lab=m, inner_lab=inn))
    return res


def all_hof_combinations_external(
        dataset_lab: str, external_nick: str, main_labs: Sequence[str], inner_labs: Sequence[str]) -> Sequence[SavedHoF]:
    res = []
    for m in main_labs:
        if not has_inner_model(m):
            res.append(default_saved_hof_from_labels_external(
                dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m, inner_lab=None))
    for inn in inner_labs:
        for m in main_labs:
            if has_inner_model(m):
                res.append(default_saved_hof_from_labels_external(
                    dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m, inner_lab=inn))
    return res


def all_ga_hofs_with_inner_model_external(
        dataset_lab: str, external_nick: str, inner_lab: Optional[str]) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=ALL_GA_LABS, inner_labs=[inner_lab])


def all_inner_hofs_for_main_external(dataset_lab: str, external_nick: str, main_lab: str) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=[main_lab], inner_labs=ALL_INNER_LABS)


def all_main_hofs_for_inner_cv(
        dataset_lab: str, inner_lab: Optional[str]) -> Sequence[SavedHoF]:
    return all_hof_combinations_cv(
        dataset_lab=dataset_lab, main_labs=ALL_MAIN, inner_labs=[inner_lab])


def all_main_hofs_for_inner_model_external(
        dataset_lab: str, external_nick: str, inner_lab: Optional[str]) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=ALL_MAIN, inner_labs=[inner_lab])


def nested_hofs_for_dataset_cv(dataset_lab: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_main_hofs_for_inner_cv(
        dataset_lab=dataset_lab, inner_lab=inner) for inner in ALL_INNER_LABS]


def nested_hofs_for_dataset_external(dataset_lab: str, external_nick: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_main_hofs_for_inner_model_external(
        dataset_lab=dataset_lab, external_nick=external_nick, inner_lab=inner) for inner in ALL_INNER_LABS]


def ga_nested_hofs_for_dataset_cv(dataset_lab: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_ga_hofs_with_inner_model_cv(
        dataset_lab=dataset_lab, inner_lab=inner) for inner in ALL_INNER_LABS]


def ga_nested_hofs_for_dataset_external(dataset_lab: str, external_nick: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_ga_hofs_with_inner_model_external(
        dataset_lab=dataset_lab, external_nick=external_nick, inner_lab=inner) for inner in ALL_INNER_LABS]


def nested_hofs_for_all_datasets_cv() -> Sequence[Sequence[Sequence[SavedHoF]]]:
    return [nested_hofs_for_dataset_cv(dataset_lab=d) for d in ALL_CV_DATASETS]


def flatten_hofs_for_dataset_cv(dataset_lab: str) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_hof_combinations_cv(
        dataset_lab=dataset_lab, main_labs=ALL_MAIN, inner_labs=ALL_INNER_LABS)


def flatten_hofs_for_dataset_external(dataset_lab: str, external_nick: str) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=ALL_MAIN, inner_labs=ALL_INNER_LABS)


TCGA_BRCA_CV_BY_INNER_MODEL = ga_nested_hofs_for_dataset_cv(dataset_lab=TCGA_BRCA_LAB)
TCGA_BRCA_EXTERNAL_BY_INNER_MODEL = nested_hofs_for_dataset_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB)
