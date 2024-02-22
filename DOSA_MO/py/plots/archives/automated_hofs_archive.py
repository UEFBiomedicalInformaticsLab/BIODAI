from typing import Optional, Sequence, Union

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from load_omics_views import MRNA_NAME
from location_manager.location_managers_archive import DEFAULT_LOCATION_MANAGER, DEFAULT_LOCATION_MANAGERS_ARCHIVE
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel, BalAccLeanness
from plots.plot_labels import TCGA_BRCA_LAB, ALL_NSGA_LABS, ALL_INNER_LABS, \
    SWEDISH_LAB, ALL_CV_DATASETS, ALL_MAIN_LABS, has_classification_inner_model, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG


def all_ga_hofs_with_inner_model_cv(dataset_lab: str, inner_lab: Optional[str],
                                    dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
                                    cox_fi: bool = True) -> Sequence[SavedHoF]:
    return [DEFAULT_LOCATION_MANAGERS_ARCHIVE.default_saved_hof_from_labels_cv(
        dataset_lab=dataset_lab, main_lab=ga, inner_lab=inner_lab,
        dir_from_label=dir_from_label, cox_fi=cox_fi) for ga in ALL_NSGA_LABS]


def all_inner_hofs_for_main_cv(dataset_lab: str, main_lab: str) -> Sequence[SavedHoF]:
    return [DEFAULT_LOCATION_MANAGER.default_saved_hof_from_labels_cv(
        dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inn) for inn in ALL_INNER_LABS]


def all_hof_combinations_cv(
        dataset_lab: str,
        main_labs: Sequence[str],
        inner_labs: Sequence[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        n_outer_folds: int = OUTER_N_FOLDS_BIG,
        cv_repeats: int = 1,
        views: Sequence[str] = (MRNA_NAME,),
        cox_fi: bool = True,
        generations: Optional[GenerationsStrategy] = None,
        hof_nick: str = PARETO_NICK,
        adjuster_regressors: Sequence[Union[None, str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS) -> Sequence[SavedHoF]:
    """Returns hofs from all location managers. They are sorted by inner label, then by main label and then by
    adjuster."""
    res = []
    for m in main_labs:
        if not has_classification_inner_model(m) or not dir_from_label.has_classification():
            for adj in adjuster_regressors:
                if adj is None or m in ALL_NSGA_LABS:  # Adjustment of non-ga is not supported yet.
                    res.extend(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_cv(
                        dataset_lab=dataset_lab, main_lab=m, inner_lab=None, dir_from_label=dir_from_label,
                        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, views=views, cox_fi=cox_fi,
                        generations=generations, hof_nick=hof_nick, adjuster_regressor=adj))
    for inn in inner_labs:
        for m in main_labs:
            if has_classification_inner_model(m) and dir_from_label.has_classification():
                for adj in adjuster_regressors:
                    res.extend(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_cv(
                        dataset_lab=dataset_lab, main_lab=m, inner_lab=inn, dir_from_label=dir_from_label,
                        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, views=views, cox_fi=cox_fi,
                        generations=generations, hof_nick=hof_nick, adjuster_regressor=adj))
    return res


def existing_hofs(hofs: Sequence[SavedHoF]) -> Sequence[SavedHoF]:
    existing_res = []
    for r in hofs:
        if r.path_exists():
            existing_res.append(r)
    return existing_res


def all_existing_hof_combinations_cv(
        dataset_lab: str,
        main_labs: Sequence[str],
        inner_labs: Sequence[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        n_outer_folds: int = OUTER_N_FOLDS_BIG,
        cv_repeats: int = 1,
        views: Sequence[str] = (MRNA_NAME,),
        cox_fi: bool = True,
        generations: Optional[GenerationsStrategy] = None,
        hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
    return existing_hofs(all_hof_combinations_cv(
        dataset_lab=dataset_lab,
        main_labs=main_labs,
        inner_labs=inner_labs,
        dir_from_label=dir_from_label,
        n_outer_folds=n_outer_folds,
        cv_repeats=cv_repeats,
        views=views,
        cox_fi=cox_fi,
        generations=generations,
        hof_nick=hof_nick))


def all_hof_combinations_external(
        dataset_lab: str, external_nick: str, main_labs: Sequence[str], inner_labs: Sequence[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        views: Sequence[str] = (MRNA_NAME,),
        cox_fi: bool = True,
        generations: Optional[GenerationsStrategy] = None,
        hof_nick: str = PARETO_NICK,
        adjuster_regressors: Sequence[Union[None, str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS) -> Sequence[SavedHoF]:
    """Returns saved hofs from all location managers."""
    res = []
    for m in main_labs:
        if not has_classification_inner_model(m) or not dir_from_label.has_classification():
            for adj in adjuster_regressors:
                if adj is None or m in ALL_NSGA_LABS:  # Adjustment of non-ga is not supported yet.
                    res.extend(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_external(
                        dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m, inner_lab=None,
                        dir_from_label=dir_from_label, views=views, cox_fi=cox_fi,
                        generations=generations, hof_nick=hof_nick, adjuster_regressor=adj))
    for inn in inner_labs:
        for m in main_labs:
            if has_classification_inner_model(m) and dir_from_label.has_classification():
                for adj in adjuster_regressors:
                    res.extend(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_external(
                        dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m, inner_lab=inn,
                        dir_from_label=dir_from_label, views=views, cox_fi=cox_fi,
                        generations=generations, hof_nick=hof_nick, adjuster_regressor=adj))
    return res


def all_ga_hofs_with_inner_model_external(
        dataset_lab: str, external_nick: str, inner_lab: Optional[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=ALL_NSGA_LABS, inner_labs=[inner_lab],
        dir_from_label=dir_from_label)


def all_inner_hofs_for_main_external(dataset_lab: str, external_nick: str, main_lab: str,
                                     dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=[main_lab], inner_labs=ALL_INNER_LABS,
        dir_from_label=dir_from_label)


def all_main_hofs_for_inner_cv(
        dataset_lab: str, inner_lab: Optional[str], main_labs=ALL_MAIN_LABS) -> Sequence[SavedHoF]:
    return all_hof_combinations_cv(
        dataset_lab=dataset_lab, main_labs=main_labs, inner_labs=[inner_lab])


def all_main_hofs_for_inner_model_external(
        dataset_lab: str, external_nick: str, inner_lab: Optional[str], main_labs=ALL_MAIN_LABS,
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=main_labs, inner_labs=[inner_lab],
        dir_from_label=dir_from_label)


def nested_hofs_for_dataset_cv(dataset_lab: str, main_labs=ALL_MAIN_LABS,
                               inner_labs: Sequence[str] = ALL_INNER_LABS) -> Sequence[Sequence[SavedHoF]]:
    """Returns a sequence for each inner label."""
    return [all_main_hofs_for_inner_cv(
        dataset_lab=dataset_lab, inner_lab=inner, main_labs=main_labs) for inner in inner_labs]


def nested_hofs_for_dataset_external(
        dataset_lab: str, external_nick: str,
        main_labs=ALL_MAIN_LABS, inner_labs=ALL_INNER_LABS,
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[Sequence[SavedHoF]]:
    return [all_main_hofs_for_inner_model_external(
        dataset_lab=dataset_lab, external_nick=external_nick, inner_lab=inner, main_labs=main_labs,
        dir_from_label=dir_from_label)
        for inner in inner_labs]


def ga_nested_hofs_for_dataset_cv(
        dataset_lab: str, dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        cox_fi: bool = True) -> Sequence[Sequence[SavedHoF]]:
    return [all_ga_hofs_with_inner_model_cv(
        dataset_lab=dataset_lab, inner_lab=inner, dir_from_label=dir_from_label,
        cox_fi=cox_fi) for inner in ALL_INNER_LABS]


def ga_nested_hofs_for_dataset_external(dataset_lab: str, external_nick: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_ga_hofs_with_inner_model_external(
        dataset_lab=dataset_lab, external_nick=external_nick, inner_lab=inner) for inner in ALL_INNER_LABS]


def nested_hofs_for_all_datasets_cv() -> Sequence[Sequence[Sequence[SavedHoF]]]:
    return [nested_hofs_for_dataset_cv(dataset_lab=d) for d in ALL_CV_DATASETS]


def flatten_hofs_for_dataset_cv(dataset_lab: str, main_labs=ALL_MAIN_LABS,
                                inner_labs: Sequence[str] = ALL_INNER_LABS,
                                dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
                                n_outer_folds: int = OUTER_N_FOLDS_BIG,
                                cv_repeats: int = 1,
                                views: Sequence[str] = (MRNA_NAME,),
                                cox_fi: bool = True,
                                generations: Optional[GenerationsStrategy] = None,
                                hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_hof_combinations_cv(
        dataset_lab=dataset_lab, main_labs=main_labs, inner_labs=inner_labs, dir_from_label=dir_from_label,
        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, views=views, cox_fi=cox_fi, generations=generations,
        hof_nick=hof_nick)


def flatten_existing_hofs_for_dataset_cv(
        dataset_lab: str, main_labs=ALL_MAIN_LABS,
        inner_labs: Sequence[str] = ALL_INNER_LABS,
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        n_outer_folds: int = OUTER_N_FOLDS_BIG,
        cv_repeats: int = 1,
        views: Sequence[str] = (MRNA_NAME,),
        cox_fi: bool = True,
        generations: Optional[GenerationsStrategy] = None,
        hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_existing_hof_combinations_cv(
        dataset_lab=dataset_lab, main_labs=main_labs, inner_labs=inner_labs, dir_from_label=dir_from_label,
        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, views=views, cox_fi=cox_fi, generations=generations,
        hof_nick=hof_nick)


def flatten_hofs_for_dataset_external(dataset_lab: str, external_nick: str, main_labs=ALL_MAIN_LABS,
                                      dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=main_labs, inner_labs=ALL_INNER_LABS,
        dir_from_label=dir_from_label)


TCGA_BRCA_CV_BY_INNER_MODEL = ga_nested_hofs_for_dataset_cv(dataset_lab=TCGA_BRCA_LAB)
TCGA_BRCA_EXTERNAL_BY_INNER_MODEL = nested_hofs_for_dataset_external(
    dataset_lab=TCGA_BRCA_LAB, external_nick=SWEDISH_LAB)
