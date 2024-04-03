from collections.abc import Sequence
from typing import Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from location_manager.basic_location_manager import BasicLocationManager
from location_manager.location_manager import LocationManager
from location_manager.location_manager_utils import OBJECTIVES_DIR_FROM_LABEL_DEFAULT, N_OUTER_FOLDS_DEFAULT, \
    CV_REPEATS_DEFAULT, VIEWS_DEFAULT, HOF_DEFAULT
from location_manager.seed_location_manager import SeedLocationManager
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel
from plots.archives.test_battery import DEFAULT_VIEW_SETS
from plots.saved_hof import SavedHoF
from setup.evaluation_setup import DEFAULT_SEED


class LocationManagersArchive:
    __location_managers: Sequence[LocationManager]

    def __init__(self, location_managers: Sequence[LocationManager]):
        """First location manager in the list is the main one used for saving."""
        self.__location_managers = list(location_managers)

    def main(self) -> LocationManager:
        return self.__location_managers[0]

    def default_saved_hof_from_labels_cv(self,
                                         dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
                                         dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
                                         n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                         cv_repeats: int = CV_REPEATS_DEFAULT,
                                         views: set[str] = VIEWS_DEFAULT,
                                         cox_fi: bool = True,
                                         generations: Optional[GenerationsStrategy] = None,
                                         hof_nick: str = HOF_DEFAULT,
                                         setup_seed: int = DEFAULT_SEED) -> SavedHoF:
        """Returns just one saved hof, from the main location manager."""
        return self.main().default_saved_hof_from_labels_cv(
            dataset_lab=dataset_lab,
            main_lab=main_lab,
            inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds,
            cv_repeats=cv_repeats,
            views=views,
            cox_fi=cox_fi,
            generations=generations,
            hof_nick=hof_nick,
            setup_seed=setup_seed)

    def all_saved_hof_from_labels_cv(self,
                                     dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
                                     dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
                                     n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                     cv_repeats: int = CV_REPEATS_DEFAULT,
                                     views: set[str] = VIEWS_DEFAULT,
                                     cox_fi: bool = True,
                                     generations: Optional[GenerationsStrategy] = None,
                                     hof_nick: str = HOF_DEFAULT,
                                     setup_seed: int = DEFAULT_SEED) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers."""
        return [lm.default_saved_hof_from_labels_cv(
            dataset_lab=dataset_lab,
            main_lab=main_lab,
            inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds,
            cv_repeats=cv_repeats,
            views=views,
            cox_fi=cox_fi,
            generations=generations,
            hof_nick=hof_nick,
            setup_seed=setup_seed)
            for lm in self.__location_managers]

    def all_seeds_hof_from_labels_cv(self,
                                     dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
                                     dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
                                     n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                     cv_repeats: int = CV_REPEATS_DEFAULT,
                                     view_sets: Sequence[set[str]] = DEFAULT_VIEW_SETS,
                                     cox_fi: bool = True,
                                     generations: Optional[GenerationsStrategy] = None,
                                     population: Optional[int] = None,
                                     hof_nick: str = HOF_DEFAULT,
                                     adjuster_regressor: Optional[str] = None
                                     ) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers."""
        res = []
        for lm in self.__location_managers:
            res.extend(lm.all_seeds_saved_hofs_from_labels_cv(
                dataset_lab=dataset_lab,
                main_lab=main_lab,
                inner_lab=inner_lab,
                dir_from_label=dir_from_label,
                n_outer_folds=n_outer_folds,
                cv_repeats=cv_repeats,
                view_sets=view_sets,
                cox_fi=cox_fi,
                generations=generations,
                population=population,
                hof_nick=hof_nick,
                adjuster_regressor=adjuster_regressor))
        return res

    def all_seeds_hof_from_labels_external(
            self,
            dataset_lab: str,
            external_nick: str,
            main_lab: str,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            view_sets: Sequence[set[str]] = DEFAULT_VIEW_SETS,
            cox_fi: bool = True,
            generations: Optional[GenerationsStrategy] = None,
            population: Optional[int] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None
            ) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers."""
        res = []
        for lm in self.__location_managers:
            res.extend(lm.all_seeds_saved_hofs_from_labels_external(
                dataset_lab=dataset_lab,
                external_nick=external_nick,
                main_lab=main_lab,
                inner_lab=inner_lab,
                dir_from_label=dir_from_label,
                view_sets=view_sets,
                cox_fi=cox_fi,
                generations=generations,
                population=population,
                hof_nick=hof_nick,
                adjuster_regressor=adjuster_regressor))
        return res

    def all_saved_hof_from_labels_external(self,
                                           dataset_lab: str, external_nick: str, main_lab: str,
                                           inner_lab: Optional[str] = None,
                                           dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
                                           views: set[str] = VIEWS_DEFAULT,
                                           cox_fi: bool = True,
                                           generations: Optional[GenerationsStrategy] = None,
                                           hof_nick: str = PARETO_NICK,
                                           adjuster_regressor: Optional[str] = None,
                                           setup_seed: int = DEFAULT_SEED) -> Sequence[SavedHoF]:
        """Returns saved hofs from all location managers."""
        return [lm.default_saved_hof_from_labels_external(
            dataset_lab=dataset_lab,
            external_nick=external_nick,
            main_lab=main_lab,
            inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views,
            cox_fi=cox_fi,
            generations=generations,
            hof_nick=hof_nick,
            adjuster_regressor=adjuster_regressor,
            setup_seed=setup_seed)
            for lm in self.__location_managers]


DEFAULT_LOCATION_MANAGERS_ARCHIVE = LocationManagersArchive(
    location_managers=[SeedLocationManager(), BasicLocationManager()])

DEFAULT_LOCATION_MANAGER = DEFAULT_LOCATION_MANAGERS_ARCHIVE.main()
