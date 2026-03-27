from collections.abc import Sequence
from typing import Optional

from sympy.core.containers import OrderedSet

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from location_manager.basic_location_manager import BasicLocationManager
from location_manager.fs_location_manager import FsLocationManager
from location_manager.location_manager import LocationManager
from location_manager.location_manager_utils import OBJECTIVES_DIR_FROM_LABEL_DEFAULT, N_OUTER_FOLDS_DEFAULT, \
    CV_REPEATS_DEFAULT, VIEWS_DEFAULT, HOF_DEFAULT, COVARIATES_DEFAULT, DEFAULT_CATEGORICAL_FI_STR, \
    DEFAULT_SURVIVAL_FI_STR
from location_manager.seed_location_manager import SeedLocationManager
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel
from plots.archives.test_battery import DEFAULT_VIEW_DEFS, DEFAULT_COVARIATE_SETS
from plots.saved_hof import SavedHoF
from setup.evaluation_setup import DEFAULT_SEED
from univariate_feature_selection.univariate_feature_selector_descriptor import \
    ManyFeatureSelectorClassDescriptor, DEFAULT_CATEGORICAL_FS_DESCRIPTOR
from views.adjusted_view_definition import AdjustedViewDef


class LocationManagersArchive:
    __location_managers: Sequence[LocationManager]

    def __init__(self, location_managers: Sequence[LocationManager]):
        """First location manager in the list is the main one used for saving."""
        self.__location_managers = list(location_managers)

    def main(self) -> LocationManager:
        return self.__location_managers[0]

    def default_saved_hof_from_labels_cv(
            self,
            dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
            cv_repeats: int = CV_REPEATS_DEFAULT,
            views: AdjustedViewDef = VIEWS_DEFAULT,
            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
            generations: Optional[GenerationsStrategy] = None,
            hof_nick: str = HOF_DEFAULT,
            setup_seed: int = DEFAULT_SEED,
            covariates: Optional[set[str]] = None,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
            ) -> Optional[SavedHoF]:
        """Returns just one saved hof, from the main location manager."""
        return self.main().default_saved_hof_from_labels_cv(
            dataset_lab=dataset_lab,
            main_lab=main_lab,
            inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds,
            cv_repeats=cv_repeats,
            views=views,
            categorical_fi=categorical_fi,
            survival_fi=survival_fi,
            generations=generations,
            hof_nick=hof_nick,
            setup_seed=setup_seed,
            covariates=covariates,
            categorical_fs_descriptor=categorical_fs_descriptor)

    def all_seeds_hof_from_labels_cv(
            self,
            dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
            cv_repeats: int = CV_REPEATS_DEFAULT,
            view_sets: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
            generations: Optional[Sequence[GenerationsStrategy]] = None,
            population: Optional[Sequence[int]] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None,
            covariate_sets: Sequence[set[str]] = DEFAULT_COVARIATE_SETS,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
            ) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers."""
        res = OrderedSet()
        for lm in self.__location_managers:
            res.update(lm.all_seeds_saved_hofs_from_labels_cv(
                dataset_lab=dataset_lab,
                main_lab=main_lab,
                inner_lab=inner_lab,
                dir_from_label=dir_from_label,
                n_outer_folds=n_outer_folds,
                cv_repeats=cv_repeats,
                view_sets=view_sets,
                categorical_fi=categorical_fi,
                survival_fi=survival_fi,
                generations=generations,
                population=population,
                hof_nick=hof_nick,
                adjuster_regressor=adjuster_regressor,
                covariate_sets=covariate_sets,
                categorical_fs_descriptor=categorical_fs_descriptor
                ))
        return list(res)

    def all_seeds_hof_from_labels_external(
            self,
            dataset_lab: str,
            external_nick: str,
            main_lab: str,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            view_sets: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
            generations: Optional[Sequence[GenerationsStrategy]] = None,
            population: Optional[Sequence[int]] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None,
            covariate_sets: Sequence[set[str]] = COVARIATES_DEFAULT,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
            ) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers.
        FDR threshold is ignored if the fs algorithm does not use it."""
        res = set()
        for lm in self.__location_managers:
            res.update(lm.all_seeds_saved_hofs_from_labels_external(
                dataset_lab=dataset_lab,
                external_nick=external_nick,
                main_lab=main_lab,
                inner_lab=inner_lab,
                dir_from_label=dir_from_label,
                view_sets=view_sets,
                categorical_fi=categorical_fi,
                survival_fi=survival_fi,
                generations=generations,
                population=population,
                hof_nick=hof_nick,
                adjuster_regressor=adjuster_regressor,
                covariate_sets=covariate_sets,
                categorical_fs_descriptor=categorical_fs_descriptor))
        return list(res)


DEFAULT_LOCATION_MANAGERS_ARCHIVE = LocationManagersArchive(
    location_managers=[FsLocationManager(), SeedLocationManager(), BasicLocationManager()])

DEFAULT_LOCATION_MANAGER = DEFAULT_LOCATION_MANAGERS_ARCHIVE.main()
