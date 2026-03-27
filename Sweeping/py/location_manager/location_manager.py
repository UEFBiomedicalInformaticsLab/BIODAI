import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

from sympy.core.containers import OrderedSet

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.hof_names import PARETO_NICK
from location_manager.location_manager_utils import (OBJECTIVES_DIR_FROM_LABEL_DEFAULT, N_OUTER_FOLDS_DEFAULT,
                                                     CV_REPEATS_DEFAULT, VIEWS_DEFAULT, HOF_DEFAULT,
                                                     optimizer_dir_from_labels_with_adjuster, hof_dir_from_label,
                                                     objectives_string, save_path_folds_str,
                                                     objectives_dir_from_label, covariates_dir, COVARIATES_DEFAULT,
                                                     DEFAULT_INNER_N_FOLDS, DEFAULT_CATEGORICAL_FI_STR,
                                                     DEFAULT_SURVIVAL_FI_STR, views_nick)
from location_manager.path_utils import create_optimizer_save_path
from objective.social_objective import PersonalObjective
from plots.archives.archives_utils import dataset_base_dir
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel
from plots.archives.test_battery import DEFAULT_VIEW_DEFS, DEFAULT_COVARIATE_SETS
from plots.plot_labels import has_inner_model, SWT_PREF, SW_PREF, NATURE_INSPIRED_MAIN
from plots.saved_hof import SavedHoF
from setup.evaluation_setup import DEFAULT_SEED
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG
from univariate_feature_selection.univariate_feature_selector_descriptor import \
    ManyFeatureSelectorClassDescriptor, DEFAULT_CATEGORICAL_FS_DESCRIPTOR, ANOVA_CATEGORICAL_DESCRIPTOR
from util.system_utils import subdirectories
from views.adjusted_view_definition import AdjustedViewDef


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
            views: AdjustedViewDef = VIEWS_DEFAULT,
            covariates: set[str] = COVARIATES_DEFAULT) -> str:
        if not has_inner_model(main_lab=main_lab):
            inner_lab = None
        path = ""
        path += dataset_base_dir(dataset_lab=dataset_lab)
        views_str = views_nick(view_to_adjusters=views)
        path += "/" + views_str + "/"
        if len(covariates) > 0:
            path += covariates_dir(covariates_names=covariates) + "/"
        path += objectives_dir_from_label(inner_lab=inner_lab, dir_from_label=dir_from_label)
        path += "/" + save_path_folds_str(outer_n_folds=n_outer_folds, cv_repeats=cv_repeats) + "/"
        return path

    def __main_path_from_labels_cv(
            self, dataset_lab: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT, cv_repeats: int = CV_REPEATS_DEFAULT,
            views: AdjustedViewDef = VIEWS_DEFAULT,
            setup_seed: int = DEFAULT_SEED,
            covariates: set[str] = COVARIATES_DEFAULT) -> str:
        """Path up to seed included, excluding optimizer."""
        path = self.__before_seed_path_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views, covariates=covariates)
        path = self._seed_adder(before_seed_path=path, seed=setup_seed)
        return path

    def __main_path_from_labels_external(
            self, dataset_lab: str, external_nick: str, main_lab: str, inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            views: AdjustedViewDef = VIEWS_DEFAULT,
            setup_seed: int = DEFAULT_SEED,
            covariates: set[str] = COVARIATES_DEFAULT) -> str:
        """Path up to seed included, excluding optimizer."""
        path = self.__before_seed_path_external(
            dataset_lab=dataset_lab, external_nick=external_nick, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views, covariates=covariates)
        path = self._seed_adder(before_seed_path=path, seed=setup_seed)
        return path

    def save_path_from_components(
            self,
            input_data_nick: str, views_to_use: AdjustedViewDef, objectives: Sequence[PersonalObjective],
            uses_inner_models: bool, outer_n_folds: int, cv_repeats: int = CV_REPEATS_DEFAULT,
            setup_seed: int = DEFAULT_SEED,
            covariate_view_names: Sequence[str] = ()) -> str:
        objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
        res = "./" + input_data_nick + "/" + views_nick(view_to_adjusters=views_to_use)
        if len(covariate_view_names) > 0:
            res += "/" + covariates_dir(covariates_names=covariate_view_names)
        res += "/" + objectives_str + "/" + save_path_folds_str(outer_n_folds=outer_n_folds,
                                                                cv_repeats=cv_repeats) + "/"
        res = self._seed_adder(before_seed_path=res, seed=setup_seed)
        return res

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
            population: Optional[int] = None,
            hof_nick: str = HOF_DEFAULT,
            adjuster_regressor: Optional[str] = None,
            setup_seed: int = DEFAULT_SEED,
            covariates: set[str] = COVARIATES_DEFAULT,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
             ) -> Optional[SavedHoF]:
        if not has_inner_model(main_lab=main_lab):
            inner_lab = None
        path = self.__main_path_from_labels_cv(
            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
            views=views,
            setup_seed=setup_seed,
            covariates=covariates
        )
        optimizer_part = self._optimizer_dir_from_labels_with_adjuster(
            main_lab=main_lab, inner_lab=inner_lab, categorical_fi=categorical_fi, survival_fi=survival_fi,
            generations=generations, population=population,
            n_outer_folds=n_outer_folds,
            adjuster_regressor=adjuster_regressor,
            categorical_fs_descriptor=categorical_fs_descriptor)
        if optimizer_part is None:
            return None
        else:
            path += optimizer_part
            path += "/hofs/"
            path += hof_dir_from_label(main_lab=main_lab, hof_nick=hof_nick)
            return SavedHoF(path=path, main_algorithm_label=main_lab, inner_lab=inner_lab,
                            adjuster_regressor=adjuster_regressor, views=views,
                            categorical_fs_descriptor=categorical_fs_descriptor,
                            covariates=covariates, dataset_lab=dataset_lab)

    @staticmethod
    def generations_for_main(main_lab: str, generations: Optional[Sequence[GenerationsStrategy]]
                             ) -> Sequence[Union[GenerationsStrategy, None]]:
        if generations is None:
            return [None]
        else:
            if SWT_PREF in main_lab:
                return [g for g in generations if g.concatenated_generations() > 0]
            elif SW_PREF in main_lab:
                return [g for g in generations if g.concatenated_generations() == 0]
            else:
                return generations

    def all_seeds_saved_hofs_from_labels_cv(
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
        res = OrderedSet()
        for views in view_sets:
            for covariates in covariate_sets:
                path = self.__before_seed_path_cv(
                    dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
                    dir_from_label=dir_from_label,
                    n_outer_folds=n_outer_folds, cv_repeats=cv_repeats,
                    views=views, covariates=covariates)
                seeds = self._seeds_to_check(before_seed_path=path)
                for g in self.generations_for_main(main_lab=main_lab, generations=generations):
                    if population is None:
                        pop = [None]
                    else:
                        pop = population
                    for p in pop:
                        to_add = [self.default_saved_hof_from_labels_cv(
                            dataset_lab=dataset_lab, main_lab=main_lab, inner_lab=inner_lab,
                            dir_from_label=dir_from_label,
                            n_outer_folds=n_outer_folds,
                            cv_repeats=cv_repeats,
                            views=views,
                            categorical_fi=categorical_fi,
                            survival_fi=survival_fi,
                            generations=g,
                            population=p,
                            hof_nick=hof_nick,
                            adjuster_regressor=adjuster_regressor,
                            setup_seed=s,
                            covariates=covariates,
                            categorical_fs_descriptor=categorical_fs_descriptor
                            )
                            for s in seeds]
                        res.update(to_add)
        if None in res:
            res.discard(None)
        return list(res)

    def all_seeds_saved_hofs_from_labels_external(
            self,
            dataset_lab: str, main_lab: str,
            external_nick: str,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
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
        res = OrderedSet()
        for views in view_sets:
            for covariates in covariate_sets:
                path = self.__before_seed_path_external(
                    dataset_lab=dataset_lab, external_nick=external_nick,
                    main_lab=main_lab, inner_lab=inner_lab,
                    dir_from_label=dir_from_label,
                    views=views, covariates=covariates)
                seeds = self._seeds_to_check(before_seed_path=path)
                for generations_choice in self.generations_for_main(main_lab=main_lab, generations=generations):
                    if population is None:
                        population = [None]
                    for pop_choice in population:
                        res.update([self.default_saved_hof_from_labels_external(
                            dataset_lab=dataset_lab, external_nick=external_nick,
                            main_lab=main_lab, inner_lab=inner_lab,
                            dir_from_label=dir_from_label,
                            views=views,
                            categorical_fi=categorical_fi,
                            survival_fi=survival_fi,
                            generations=generations_choice,
                            population=pop_choice,
                            hof_nick=hof_nick,
                            adjuster_regressor=adjuster_regressor,
                            setup_seed=s, covariates=covariates,
                            categorical_fs_descriptor=categorical_fs_descriptor)
                            for s in seeds])
        if None in res:
            res.discard(None)
        return list(res)

    def optimizer_save_path(
            self,
            input_data_nick: str, views_to_use: AdjustedViewDef, objectives: Sequence[PersonalObjective],
            uses_inner_models: bool, outer_n_folds: int,
            optimizer_nick: str,
            cv_repeats: int = CV_REPEATS_DEFAULT,
            setup_seed: int = DEFAULT_SEED,
            covariate_view_names: Sequence[str] = ()
            ) -> str:
        """Ends with '/'."""
        return create_optimizer_save_path(
            save_path=self.save_path_from_components(
                input_data_nick=input_data_nick, views_to_use=views_to_use, objectives=objectives,
                uses_inner_models=uses_inner_models, outer_n_folds=outer_n_folds, cv_repeats=cv_repeats,
                setup_seed=setup_seed, covariate_view_names=covariate_view_names),
            optimizer_nick=optimizer_nick)

    @staticmethod
    def __before_seed_path_external_from_strings(
            input_data_nick: str, views_to_use: AdjustedViewDef, objectives: Sequence[PersonalObjective],
            uses_inner_models: bool, external_data_nick: str, covariates: set[str] = COVARIATES_DEFAULT) -> str:
        objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
        res = "./" + input_data_nick + "/" + views_nick(view_to_adjusters=views_to_use)
        if len(covariates) > 0:
            res += "/" + covariates_dir(covariates_names=covariates)
        res += "/" + objectives_str + "/external_validation/" + external_data_nick + "/"
        return res

    def save_path_external_from_strings(
            self,
            input_data_nick: str, views_to_use: AdjustedViewDef, objectives: Sequence[PersonalObjective],
            uses_inner_models: bool, external_data_nick: str, setup_seed: int = DEFAULT_SEED,
            covariates: set[str] = COVARIATES_DEFAULT) -> str:
        res = self.__before_seed_path_external_from_strings(
            input_data_nick=input_data_nick, views_to_use=views_to_use, objectives=objectives,
            uses_inner_models=uses_inner_models, external_data_nick=external_data_nick,
            covariates=covariates)
        res = self._seed_adder(before_seed_path=res, seed=setup_seed)
        return res

    @staticmethod
    def __before_seed_path_external(
            dataset_lab: str, external_nick: str,
            main_lab: str,
            views: AdjustedViewDef = VIEWS_DEFAULT,
            inner_lab: Optional[str] = None,
            dir_from_label: ObjectivesDirFromLabel = OBJECTIVES_DIR_FROM_LABEL_DEFAULT,
            covariates: set[str] = COVARIATES_DEFAULT) -> str:
        if not has_inner_model(main_lab=main_lab):
            inner_lab = None
        path = ""
        path += dataset_base_dir(dataset_lab=dataset_lab)
        path += "/" + views_nick(view_to_adjusters=views) + "/"
        if len(covariates) > 0:
            path += covariates_dir(covariates_names=covariates) + "/"
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
            views: AdjustedViewDef = VIEWS_DEFAULT,
            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
            generations: Optional[GenerationsStrategy] = None,
            population: Optional[int] = None,
            hof_nick: str = PARETO_NICK,
            adjuster_regressor: Optional[str] = None,
            setup_seed: int = DEFAULT_SEED,
            covariates: set[str] = COVARIATES_DEFAULT,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
            ) -> Optional[SavedHoF]:

        if not has_inner_model(main_lab=main_lab):
            inner_lab = None
        path = self.__main_path_from_labels_external(
            dataset_lab=dataset_lab, external_nick=external_nick, main_lab=main_lab, inner_lab=inner_lab,
            dir_from_label=dir_from_label,
            views=views,
            setup_seed=setup_seed,
            covariates=covariates
        )
        optimizer_dir = self._optimizer_dir_from_labels_with_adjuster(
            main_lab=main_lab, inner_lab=inner_lab, categorical_fi=categorical_fi, survival_fi=survival_fi,
            generations=generations, population=population, n_outer_folds=OUTER_N_FOLDS_BIG,
            adjuster_regressor=adjuster_regressor,
            categorical_fs_descriptor=categorical_fs_descriptor)
        if optimizer_dir is None:
            return None
        else:
            path += optimizer_dir
            # A hypothetical outer_n_folds is needed for the adjusted algorithm.
            path += "/hofs/"
            path += hof_dir_from_label(main_lab=main_lab, hof_nick=hof_nick)
            return SavedHoF(path=path, main_algorithm_label=main_lab, inner_lab=inner_lab,
                            adjuster_regressor=adjuster_regressor, views=views,
                            categorical_fs_descriptor=categorical_fs_descriptor, covariates=covariates,
                            dataset_lab=dataset_lab+"-"+external_nick)

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

    def _optimizer_dir_from_labels_with_adjuster(self,
            main_lab: str, inner_lab: Optional[str],
            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
            generations: Optional[GenerationsStrategy] = None,
            population: Optional[int] = None,
            adjuster_regressor: Optional[str] = None,
            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
            inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
            ) -> Optional[str]:
        """Only adjusted optimizers with GA as tuning algorithm and no sweep generations are supported."""
        fs_string = self.fs_string_if_needed(
            main_lab=main_lab, categorical_fs_descriptor=categorical_fs_descriptor)
        if fs_string is None:
            return None
        else:
            return optimizer_dir_from_labels_with_adjuster(
                main_lab=main_lab, inner_lab=inner_lab,
                categorical_fi=categorical_fi,
                survival_fi=survival_fi, generations=generations, population=population,
                adjuster_regressor=adjuster_regressor, n_outer_folds=n_outer_folds, inner_n_folds=inner_n_folds
            ) + fs_string

    def fs_string_if_needed(
            self,
            main_lab: str,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
    ) -> Optional[str]:
        """Only nature inspired methods use univariate fs at the moment."""
        if main_lab in NATURE_INSPIRED_MAIN:
            return self._fs_string(categorical_fs_descriptor=categorical_fs_descriptor)
        else:
            return ""

    def _fs_string(
            self,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
    ) -> Optional[str]:
        """Returns None if the fs method is not supported by this location manager.
        This default implementation accepts only ANOVA, that has been the only used method before 2025,
        and returns the empty string."""
        if categorical_fs_descriptor == ANOVA_CATEGORICAL_DESCRIPTOR:
            return ""
        else:
            return None
