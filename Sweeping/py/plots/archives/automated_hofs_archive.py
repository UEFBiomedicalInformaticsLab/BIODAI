from collections.abc import Iterable
from typing import Optional, Sequence, Union

from orderedset import OrderedSet

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.hof_names import PARETO_NICK
from location_manager.location_manager_utils import DEFAULT_CATEGORICAL_FIS, DEFAULT_CATEGORICAL_FI_STR, \
    DEFAULT_SURVIVAL_FIS, COX_FI_STR, DEFAULT_SURVIVAL_FI_STR
from location_manager.location_managers_archive import DEFAULT_LOCATION_MANAGER, DEFAULT_LOCATION_MANAGERS_ARCHIVE
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel, BalAccLeanness
from plots.archives.test_battery import DEFAULT_VIEW_DEFS, DEFAULT_COVARIATE_SETS, DEFAULT_CATEGORICAL_FS_DESCRIPTORS
from plots.plot_labels import (ALL_NSGA_LABS, ALL_INNER_LABS, ALL_CV_DATASETS, ALL_MAIN_LABS, has_inner_model,
                               DEFAULT_ADJUSTER_REGRESSORS_LABS)
from plots.saved_hof import SavedHoF
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG
from univariate_feature_selection.univariate_feature_selector_descriptor import ManyFeatureSelectorClassDescriptor
from util.str_utils import str_in_lines
from views.adjusted_view_definition import AdjustedViewDef


def all_ga_hofs_with_inner_model_cv(dataset_lab: str, inner_lab: Optional[str],
                                    dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
                                    categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
                                    survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
                                    covariates: Optional[set[str]] = None) -> Sequence[SavedHoF]:
    res = set()
    for ga in ALL_NSGA_LABS:
        res.update(DEFAULT_LOCATION_MANAGERS_ARCHIVE.default_saved_hof_from_labels_cv(
            dataset_lab=dataset_lab, main_lab=ga, inner_lab=inner_lab,
            dir_from_label=dir_from_label, categorical_fi=categorical_fi, survival_fi=survival_fi,
            covariates=covariates))
    res.discard(None)
    return list(res)


def all_inner_hofs_for_main_cv(dataset_lab: str, main_lab: str) -> Sequence[SavedHoF]:
    res = set()
    res.update(DEFAULT_LOCATION_MANAGER.default_saved_hof_from_labels_cv(
        dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inn, survival_fi=COX_FI_STR) for inn in ALL_INNER_LABS)
    res.discard(None)
    return list(res)


def all_hof_combinations_cv(
        dataset_lab: str,
        main_labs: Sequence[str],
        inner_labs: Sequence[str] = ALL_INNER_LABS,
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        n_outer_folds: int = OUTER_N_FOLDS_BIG,
        cv_repeats: int = 1,
        view_sets: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
        categorical_fis: Sequence[str] = DEFAULT_CATEGORICAL_FIS,
        survival_fis: Sequence[str] = DEFAULT_SURVIVAL_FIS,
        generations: Optional[Sequence[GenerationsStrategy]] = None,
        population: Optional[Sequence[int]] = None,
        hof_nick: str = PARETO_NICK,
        adjuster_regressors: Sequence[Union[None, str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
        covariate_sets: Sequence[set[str]] = DEFAULT_COVARIATE_SETS,
        categorical_fs_descriptors: Iterable[ManyFeatureSelectorClassDescriptor] = DEFAULT_CATEGORICAL_FS_DESCRIPTORS,
        verbose: bool = False
) -> Sequence[SavedHoF]:
    """Returns hofs from all location managers. They are sorted by inner label, then by main label and then by
    adjuster."""
    res = OrderedSet()
    for m in main_labs:
        if not has_inner_model(m) or not dir_from_label.has_classification():
            for adj in adjuster_regressors:
                if adj is None or m in ALL_NSGA_LABS:  # Adjustment of non-ga is not supported yet.
                    for covs in covariate_sets:  # This way HoFs are grouped by covariates then by FS, FDR, etc.
                        for categorical_fs_descriptor in categorical_fs_descriptors:
                            for categorical_fi in categorical_fis:
                                for survival_fi in survival_fis:
                                    res.update(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_cv(
                                        dataset_lab=dataset_lab, main_lab=m, inner_lab=None,
                                        dir_from_label=dir_from_label,
                                        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, view_sets=view_sets,
                                        categorical_fi=categorical_fi, survival_fi=survival_fi,
                                        generations=generations, population=population, hof_nick=hof_nick,
                                        adjuster_regressor=adj, covariate_sets=[covs],
                                        categorical_fs_descriptor=categorical_fs_descriptor))
    for inn in inner_labs:
        for m in main_labs:
            if has_inner_model(m) and dir_from_label.has_classification():
                for adj in adjuster_regressors:
                    for covs in covariate_sets:  # This way HoFs are grouped by covariates then by FS, FDR, etc.
                        for categorical_fs_descriptor in categorical_fs_descriptors:
                            for categorical_fi in categorical_fis:
                                for survival_fi in survival_fis:
                                    res.update(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_cv(
                                        dataset_lab=dataset_lab, main_lab=m, inner_lab=inn,
                                        dir_from_label=dir_from_label,
                                        n_outer_folds=n_outer_folds, cv_repeats=cv_repeats, view_sets=view_sets,
                                        categorical_fi=categorical_fi, survival_fi=survival_fi,
                                        generations=generations, population=population, hof_nick=hof_nick,
                                        adjuster_regressor=adj, covariate_sets=[covs],
                                        categorical_fs_descriptor=categorical_fs_descriptor))
    res = list(res)
    if verbose:
        print(str_in_lines(li=res))
    return res


def all_hof_combinations_external(
        dataset_lab: str, external_nick: str, main_labs: Sequence[str], inner_labs: Sequence[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        view_sets: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
        categorical_fis: Sequence[str] = DEFAULT_CATEGORICAL_FIS,
        survival_fis: Sequence[str] = DEFAULT_SURVIVAL_FIS,
        generations: Optional[Sequence[GenerationsStrategy]] = None,
        population: Optional[Sequence[int]] = None,
        hof_nick: str = PARETO_NICK,
        adjuster_regressors: Sequence[Union[None, str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
        covariate_sets: Sequence[set[str]] = DEFAULT_COVARIATE_SETS,
        categorical_fs_descriptors: Iterable[ManyFeatureSelectorClassDescriptor] = DEFAULT_CATEGORICAL_FS_DESCRIPTORS
        ) -> Sequence[SavedHoF]:
    """Returns saved hofs from all location managers."""
    res = OrderedSet()
    for m in main_labs:
        if not has_inner_model(m) or not dir_from_label.has_classification():
            for adj in adjuster_regressors:
                if adj is None or m in ALL_NSGA_LABS:  # Adjustment of non-ga is not supported yet.
                    for covs in covariate_sets:  # This way HoFs are grouped by covariates then by FS, FDR, etc.
                        for categorical_fs_descriptor in categorical_fs_descriptors:
                            for categorical_fi in categorical_fis:
                                for survival_fi in survival_fis:
                                    res.update(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_external(
                                        dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m,
                                        inner_lab=None, dir_from_label=dir_from_label, view_sets=view_sets,
                                        categorical_fi=categorical_fi, survival_fi=survival_fi,
                                        generations=generations, population=population, hof_nick=hof_nick,
                                        adjuster_regressor=adj, covariate_sets=[covs],
                                        categorical_fs_descriptor=categorical_fs_descriptor))
    for inn in inner_labs:
        for m in main_labs:
            if has_inner_model(m) and dir_from_label.has_classification():
                for adj in adjuster_regressors:
                    for covs in covariate_sets:  # This way HoFs are grouped by covariates then by FS, FDR, etc.
                        for categorical_fs_descriptor in categorical_fs_descriptors:
                                for categorical_fi in categorical_fis:
                                    for survival_fi in survival_fis:
                                        res.update(DEFAULT_LOCATION_MANAGERS_ARCHIVE.all_seeds_hof_from_labels_external(
                                            dataset_lab=dataset_lab, external_nick=external_nick, main_lab=m,
                                            inner_lab=inn, dir_from_label=dir_from_label, view_sets=view_sets,
                                            categorical_fi=categorical_fi, survival_fi=survival_fi,
                                            generations=generations, population=population, hof_nick=hof_nick,
                                            adjuster_regressor=adj, covariate_sets=[covs],
                                            categorical_fs_descriptor=categorical_fs_descriptor))
    return list(res)


def existing_hofs(hofs: Sequence[SavedHoF], verbose: bool = False) -> Sequence[SavedHoF]:
    existing_res = []
    for r in hofs:
        if r.path_exists():
            existing_res.append(r)
        else:
            if verbose:
                print("Not found " + str(r.path()))
    return existing_res


def all_existing_hof_combinations_cv(
        dataset_lab: str,
        main_labs: Sequence[str],
        inner_labs: Sequence[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
        n_outer_folds: int = OUTER_N_FOLDS_BIG,
        cv_repeats: int = 1,
        view_sets: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
        categorical_fi_nicks: Sequence[str] = DEFAULT_CATEGORICAL_FIS,
        survival_fis: Sequence[str] = DEFAULT_SURVIVAL_FIS,
        generations: Optional[Sequence[GenerationsStrategy]] = None,
        population: Optional[Sequence[int]] = None,
        hof_nick: str = PARETO_NICK,
        adjuster_regressors: Sequence[Optional[str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
        covariate_sets: Sequence[set[str]] = DEFAULT_COVARIATE_SETS,
        categorical_fs_descriptors: Iterable[ManyFeatureSelectorClassDescriptor] = DEFAULT_CATEGORICAL_FS_DESCRIPTORS
) -> Sequence[SavedHoF]:
    return existing_hofs(all_hof_combinations_cv(
        dataset_lab=dataset_lab,
        main_labs=main_labs,
        inner_labs=inner_labs,
        dir_from_label=dir_from_label,
        n_outer_folds=n_outer_folds,
        cv_repeats=cv_repeats,
        view_sets=view_sets,
        categorical_fis=categorical_fi_nicks,
        survival_fis=survival_fis,
        generations=generations,
        population=population,
        hof_nick=hof_nick,
        adjuster_regressors=adjuster_regressors,
        covariate_sets=covariate_sets,
        categorical_fs_descriptors=categorical_fs_descriptors))


def all_ga_hofs_with_inner_model_external(
        dataset_lab: str, external_nick: str, inner_lab: Optional[str],
        dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=ALL_NSGA_LABS, inner_labs=[inner_lab],
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


def ga_nested_hofs_for_dataset_external(dataset_lab: str, external_nick: str) -> Sequence[Sequence[SavedHoF]]:
    return [all_ga_hofs_with_inner_model_external(
        dataset_lab=dataset_lab, external_nick=external_nick, inner_lab=inner) for inner in ALL_INNER_LABS]


def nested_hofs_for_all_datasets_cv() -> Sequence[Sequence[Sequence[SavedHoF]]]:
    return [nested_hofs_for_dataset_cv(dataset_lab=d) for d in ALL_CV_DATASETS]


def flatten_hofs_for_dataset_external(dataset_lab: str, external_nick: str, main_labs=ALL_MAIN_LABS,
                                      dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> Sequence[SavedHoF]:
    """Returns a sequence of SavedHoF, one for each combination of main algorithm and inner algorithm."""
    return all_hof_combinations_external(
        dataset_lab=dataset_lab, external_nick=external_nick, main_labs=main_labs, inner_labs=ALL_INNER_LABS,
        dir_from_label=dir_from_label)
