import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from load_omics_views import MRNA_NAME
from location_manager.location_manager_utils import (OBJECTIVES_DIR_FROM_LABEL_DEFAULT, N_OUTER_FOLDS_DEFAULT,
                                                     CV_REPEATS_DEFAULT, VIEWS_DEFAULT, HOF_DEFAULT,
                                                     optimizer_dir_from_labels_with_adjuster, hof_dir_from_label,
                                                     objectives_string, views_string, save_path_folds_str,
                                                     objectives_dir_from_label)
from location_manager.path_utils import create_optimizer_save_path
from objective.social_objective import PersonalObjective
from plots.archives.archives_utils import dataset_base_dir
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel
from plots.plot_labels import has_classification_inner_model, main_and_inner_label
from plots.saved_hof import SavedHoF
from setup.evaluation_setup import DEFAULT_SEED
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG
from util.system_utils import subdirectories


class LocationManager(ABC):

    @abstractmethod
    def _seed_adder(self, before_seed_path: str, seed: int) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _seeds_to_check(self, before_seed_path: str) -> Sequence[int]:
        raise NotImplementedError()

    @staticmethod
    def __before_seed_path_cv(
            dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT, cv_repeats: int = CV_REPEATS_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT) -> str:
        if not has_classification_inner_model(main_lab=main_lab):
            inner_lab = None
        path = ""
        path += dataset_base_dir(dataset_lab=dataset_lab)
        views_str = views_string(view_names=views)
        path += "/" + views_str + "/"
        path += objectives_dir_from_label(inner_lab=inner_lab, dir_from_label=dir_from_label)
        path += "/" + save_path_folds_str(outer_n_folds=n_outer_folds, cv_repeats=cv_repeats) + "/"
        return path

    def __main_path_from_labels_cv(
            self, dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT, cv_repeats: int = CV_REPEATS_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT,
            setup_seed: int = DEFAULT_SEED) -> str:
        """Path up to seed included, excluding optimizer."""
        path = self.__before_seed_path_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views)
        path = self._seed_adder(before_seed_path=path, seed=setup_seed)
        return path

    def __main_path_from_labels_external(
            self, dataset_lab: str, external_nick: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT,
            setup_seed: int = DEFAULT_SEED) -> str:
        """Path up to seed included, excluding optimizer."""
        path = self.__before_seed_path_external(
            dataset_lab=dataset_lab, external_nick=external_nick, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views)
        path = self._seed_adder(before_seed_path=path, seed=setup_seed)
        return path

    def save_path_from_components(
            self,
            input_data_nick: str, views_to_use: Sequence[str], objectives: [PersonalObjective],
            uses_inner_models: bool, outer_n_folds: int, cv_repeats: int = CV_REPEATS_DEFAULT,
            setup_seed: int = DEFAULT_SEED) -> str:
        objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
        res = "./" + input_data_nick + "/" + views_string(views_to_use)
        res += "/" + objectives_str + "/" + save_path_folds_str(outer_n_folds=outer_n_folds,
                                                                cv_repeats=cv_repeats) + "/"
        res = self._seed_adder(before_seed_path=res, seed=setup_seed)
        return res

    def default_saved_hof_from_labels_cv(self,
                                         dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
                                         dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
                                         n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                         cv_repeats: int = CV_REPEATS_DEFAULT,
                                         views: Sequence[str] = VIEWS_DEFAULT,
                                         cox_fi: bool = True,
                                         generations: Optional[GenerationsStrategy] = None,
                                         hof_nick: str = HOF_DEFAULT,
                                         adjuster_regressor: Optional[str] = None,
                                         setup_seed: int = DEFAULT_SEED) -> SavedHoF:
        if not has_classification_inner_model(main_lab=main_lab):
            inner_lab = None
        name = main_and_inner_label(main_lab=main_lab, inner_lab=inner_lab, adjuster_regressor=adjuster_regressor)
        path = self.__main_path_from_labels_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views,
            setup_seed=setup_seed
        )
        path += optimizer_dir_from_labels_with_adjuster(main_lab=main_lab, inner_lab=inner_lab, cox_fi=cox_fi,
                                                        generations=generations, n_outer_folds=n_outer_folds,
                                                        adjuster_regressor=adjuster_regressor)
        path += "/hofs/"
        path += hof_dir_from_label(main_lab=main_lab, hof_nick=hof_nick)
        return SavedHoF(name=name, path=path, main_algorithm_label=main_lab)

    def all_seeds_saved_hofs_from_labels_cv(
            self,
            dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
            cv_repeats: int = CV_REPEATS_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT,
            cox_fi: bool = True,
            generations: Optional[GenerationsStrategy] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None) -> Sequence[SavedHoF]:
        path = self.__before_seed_path_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views)
        seeds = self._seeds_to_check(before_seed_path=path)
        return [self.default_saved_hof_from_labels_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds,
            cv_repeats=cv_repeats,
            views=views,
            cox_fi=cox_fi,
            generations=generations,
            hof_nick=hof_nick,
            adjuster_regressor=adjuster_regressor,
            setup_seed=s)
            for s in seeds]

    def all_seeds_saved_hofs_from_labels_external(
            self,
            dataset_lab: str, main_lab: str,
            external_nick: str,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT,
            cox_fi: bool = True,
            generations: Optional[GenerationsStrategy] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None) -> Sequence[SavedHoF]:
        path = self.__before_seed_path_external(
            dataset_lab=dataset_lab, external_nick=external_nick,
            main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views)
        seeds = self._seeds_to_check(before_seed_path=path)
        return [self.default_saved_hof_from_labels_external(
            dataset_lab=dataset_lab, external_nick=external_nick,
            main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views,
            cox_fi=cox_fi,
            generations=generations,
            hof_nick=hof_nick,
            adjuster_regressor=adjuster_regressor,
            setup_seed=s)
            for s in seeds]

    def optimizer_save_path(
            self,
            input_data_nick: str, views_to_use: Sequence[str], objectives: [PersonalObjective],
            uses_inner_models: bool, outer_n_folds: int,
            optimizer_nick: str,
            cv_repeats: int = CV_REPEATS_DEFAULT,
            setup_seed: int = DEFAULT_SEED
            ) -> str:
        """Ends with '/'."""
        return create_optimizer_save_path(
            save_path=self.save_path_from_components(
                input_data_nick=input_data_nick, views_to_use=views_to_use, objectives=objectives,
                uses_inner_models=uses_inner_models, outer_n_folds=outer_n_folds, cv_repeats=cv_repeats,
                setup_seed=setup_seed),
            optimizer_nick=optimizer_nick)

    @staticmethod
    def __before_seed_path_external_from_strings(
            input_data_nick: str, views_to_use: [str], objectives: [PersonalObjective],
            uses_inner_models: bool, external_data_nick: str) -> str:
        objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
        res = "./" + input_data_nick + "/" + views_string(views_to_use)
        res += "/" + objectives_str + "/external_validation/" + external_data_nick + "/"
        return res

    def save_path_external_from_strings(
            self,
            input_data_nick: str, views_to_use: [str], objectives: [PersonalObjective],
            uses_inner_models: bool, external_data_nick: str, setup_seed: int = DEFAULT_SEED) -> str:
        res = self.__before_seed_path_external_from_strings(
            input_data_nick=input_data_nick, views_to_use=views_to_use, objectives=objectives,
            uses_inner_models=uses_inner_models, external_data_nick=external_data_nick)
        res = self._seed_adder(before_seed_path=res, seed=setup_seed)
        return res

    @staticmethod
    def __before_seed_path_external(
            dataset_lab: str, external_nick: str,
            main_lab: str,
            views: Sequence[str] = VIEWS_DEFAULT,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT) -> str:
        if not has_classification_inner_model(main_lab=main_lab):
            inner_lab = None
        path = ""
        path += dataset_base_dir(dataset_lab=dataset_lab)
        path += "/" + views_string(view_names=views) + "/"
        path += objectives_dir_from_label(inner_lab=inner_lab, dir_from_label=dir_from_label)
        path += "/external_validation/"
        path += external_nick
        path += "/"
        return path

    def default_saved_hof_from_labels_external(
            self,
            dataset_lab: str, external_nick: str, main_lab: str,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            views: Sequence[str] = (MRNA_NAME,),
            cox_fi: bool = True,
            generations: Optional[GenerationsStrategy] = None,
            hof_nick: str = PARETO_NICK,
            adjuster_regressor: Optional[str] = None,
            setup_seed: int = DEFAULT_SEED) -> SavedHoF:

        if not has_classification_inner_model(main_lab=main_lab):
            inner_lab = None
        name = main_and_inner_label(main_lab=main_lab, inner_lab=inner_lab, adjuster_regressor=adjuster_regressor)
        path = self.__main_path_from_labels_external(
            dataset_lab=dataset_lab, external_nick=external_nick, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views,
            setup_seed=setup_seed
        )
        optimizer_dir = optimizer_dir_from_labels_with_adjuster(
            main_lab=main_lab, inner_lab=inner_lab, cox_fi=cox_fi,
            generations=generations, n_outer_folds=OUTER_N_FOLDS_BIG,
            adjuster_regressor=adjuster_regressor)
        path += optimizer_dir
        # A hypothetical outer_n_folds is needed for the adjusted algorithm.
        path += "/hofs/"
        path += hof_dir_from_label(main_lab=main_lab, hof_nick=hof_nick)
        return SavedHoF(name=name, path=path, main_algorithm_label=main_lab)

    @staticmethod
    def _seed_directories_from_path(
            before_seed_path: str) -> list[int]:
        subdirs = subdirectories(main_directory=before_seed_path)
        res = []
        for s in subdirs:
            s = os.path.basename(os.path.normpath(s))
            if s.isdigit():
                res.append(int(s))
        return res

    def _seed_directories(
            self, dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT, cv_repeats: int = CV_REPEATS_DEFAULT,
            views: Sequence[str] = VIEWS_DEFAULT) -> list[int]:
        path = self.__before_seed_path_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views)
        return self._seed_directories_from_path(before_seed_path=path)
