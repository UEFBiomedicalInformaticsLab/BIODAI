from collections.abc import Sequence, Iterable
from typing import Optional

from cross_validation.multi_objective.optimizer.ga_str_utils import nick_paste
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.hof_names import PARETO_NICK
from location_manager.location_manager_utils import DEFAULT_CATEGORICAL_FIS, DEFAULT_SURVIVAL_FIS
from objective.objective_computer import ObjectiveComputer
from plots.archives.archives_utils import dataset_base_dir
from plots.archives.automated_hofs_archive import all_hof_combinations_external, existing_hofs
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabelByComputers
from plots.archives.optimizer_descriptor import OptimizerDescriptor
from plots.archives.test_battery import TestBattery, DEFAULT_VIEW_DEFS, DEFAULT_COVARIATE_SETS, \
    DEFAULT_CATEGORICAL_FS_DESCRIPTORS
from plots.hofs_plotter.plot_setup import PlotSetup, PlotSetupWithDefaultLabels
from plots.plot_labels import ALL_MAIN_LABS, ALL_INNER_LABS, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF
from univariate_feature_selection.univariate_feature_selector_descriptor import ManyFeatureSelectorClassDescriptor
from util.named import NickNamed
from util.sequence_utils import clean_redundant_subsequences, transpose
from util.str_utils import iterable_to_string
from views.adjusted_view_definition import AdjustedViewDef


class ExternalValidationDatasets(NickNamed):
    __internal_dataset_label: str
    __external_dataset_label: str

    def __init__(self, internal_label: str, external_label: str):
        self.__internal_dataset_label = internal_label
        self.__external_dataset_label = external_label

    def internal_dataset_label(self) -> str:
        return self.__internal_dataset_label

    def external_dataset_label(self) -> str:
        return self.__external_dataset_label

    def nick(self) -> str:
        return self.__internal_dataset_label + "_" + self.__external_dataset_label

    def internal_dataset_nick(self) -> str:
        return dataset_base_dir(dataset_lab=self.internal_dataset_label())

    def external_dataset_nick(self) -> str:
        return dataset_base_dir(dataset_lab=self.external_dataset_label())

    def __str__(self) -> str:
        return self.__internal_dataset_label + " - " + self.__external_dataset_label

    def name(self) -> str:
        return str(self)


class TestBatteryExternal(TestBattery):
    __datasets: Sequence[ExternalValidationDatasets]

    def __init__(self,
                 objective_computers: Sequence[ObjectiveComputer],
                 datasets: Iterable[ExternalValidationDatasets],
                 view_defs: Sequence[AdjustedViewDef] = DEFAULT_VIEW_DEFS,
                 main_labs: Sequence[str] = ALL_MAIN_LABS,
                 generations: Optional[Sequence[GenerationsStrategy]] = None,
                 population: Optional[Sequence[int]] = None,
                 inner_labs: Sequence[str] = ALL_INNER_LABS,
                 categorical_fi_nicks: Sequence[str] = DEFAULT_CATEGORICAL_FIS,
                 survival_fi_nicks: Sequence[str] = DEFAULT_SURVIVAL_FIS,
                 adjuster_regressors: Sequence[Optional[str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
                 nick: Optional[str] = None,
                 plot_setup: PlotSetup = PlotSetupWithDefaultLabels(),
                 baseline: Optional[OptimizerDescriptor] = None,
                 covariate_sets: Sequence[set[str]] = DEFAULT_COVARIATE_SETS,
                 categorical_fs_descriptors: Iterable[
                     ManyFeatureSelectorClassDescriptor] = DEFAULT_CATEGORICAL_FS_DESCRIPTORS
                 ):
        self.__datasets = list(datasets)
        TestBattery.__init__(self,
                             objective_computers=objective_computers,
                             view_defs=view_defs,
                             main_labs=main_labs,
                             generations=generations,
                             population=population,
                             inner_labs=inner_labs,
                             categorical_fi_nicks=categorical_fi_nicks,
                             survival_fi_nicks=survival_fi_nicks,
                             adjuster_regressors=adjuster_regressors,
                             nick=nick,
                             plot_setup=plot_setup,
                             baseline=baseline,
                             covariate_sets=covariate_sets,
                             categorical_fs_descriptors=categorical_fs_descriptors)

    def datasets(self) -> Sequence[ExternalValidationDatasets]:
        return self.__datasets

    def _automatic_nick(self) -> str:
        return nick_paste(parts=[d.nick() for d in self.__datasets])

    def is_external(self) -> bool:
        return True

    def n_validations(self) -> int:
        return len(self.__datasets)

    def existing_flat_hofs_for_datasets(
            self, datasets: ExternalValidationDatasets, hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
        return existing_hofs(all_hof_combinations_external(
            dataset_lab=datasets.internal_dataset_label(),
            external_nick=datasets.external_dataset_nick(),
            main_labs=self.main_labs(),
            inner_labs=self.inner_labs(),
            dir_from_label=self.dir_from_label(),
            view_sets=self.view_defs(),
            categorical_fis=self.categorical_fi_nicks(),
            survival_fis=self.survival_fi_nicks(),
            generations=self.generations(),
            population=self.population(),
            hof_nick=hof_nick,
            adjuster_regressors=self.adjuster_regressors(),
            covariate_sets=self.covariate_sets(),
            categorical_fs_descriptors=self.categorical_fs_descriptors()))

    def datasets_report_path_part(self, datasets: ExternalValidationDatasets) -> str:
        battery_nick = self.nick()
        objective_computers = self.objective_computers()
        single_dataset = self.n_validations() < 2
        if battery_nick is None:
            dataset_path_part = datasets.nick()
        else:
            if single_dataset:
                dataset_path_part = battery_nick
            else:
                dataset_path_part = battery_nick + "_" + datasets.nick()
        objectives_str = iterable_to_string(sorted([o.nick() for o in objective_computers]),
                                            compact=True, separator="_", brackets=False)
        return dataset_path_part + "/" + objectives_str

    def existing_nested_hofs_for_datasets(
            self, datasets: ExternalValidationDatasets, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """A list of sequences, one for each inner model."""
        dir_from_label = self.dir_from_label()
        res = []
        for inner in self.inner_labs():
            hofs = all_hof_combinations_external(
                main_labs=self.main_labs(),
                view_sets=self.view_defs(),
                dir_from_label=dir_from_label,
                dataset_lab=datasets.internal_dataset_label(),
                external_nick=datasets.external_dataset_nick(),
                inner_labs=[inner],
                categorical_fis=self.categorical_fi_nicks(),
                survival_fis=self.survival_fi_nicks(),
                generations=self.generations(),
                population=self.population(),
                hof_nick=hof_nick,
                adjuster_regressors=self.adjuster_regressors(),
                covariate_sets=self.covariate_sets(),
                categorical_fs_descriptors=self.categorical_fs_descriptors())
            existing = existing_hofs(hofs)
            if len(existing) > 0:
                res.append(existing)
        return clean_redundant_subsequences(res)

    def existing_hofs_grouped_by_dataset_and_inner(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        res = []
        for d in self.datasets():
            res.extend(self.existing_nested_hofs_for_datasets(datasets=d, hof_nick=hof_nick))
        return res

    def existing_flat_hofs(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        res = []
        for d in self.datasets():
            res.append(self.existing_flat_hofs_for_datasets(datasets=d, hof_nick=hof_nick))
        return res

    def existing_hofs_grouped_by_inner_and_dataset(self, hof_nick: str = PARETO_NICK
                                                   ) -> list[Sequence[Sequence[SavedHoF]]]:
        res = []
        for d in self.datasets():
            res.append(self.existing_nested_hofs_for_datasets(datasets=d, hof_nick=hof_nick))
        return transpose(res)

    def baseline_hofs(self, hof_nick: str = PARETO_NICK) -> Sequence[Sequence[SavedHoF]]:
        dir_from_label = ObjectivesDirFromLabelByComputers(objectives=self.objective_computers())
        baseline = self.baseline()
        return [existing_hofs(hofs=all_hof_combinations_external(
            main_labs=[baseline.main_lab()],
            view_sets=[baseline.view_set()],
            dir_from_label=dir_from_label,
            dataset_lab=datasets.internal_dataset_label(),
            external_nick=datasets.external_dataset_nick(),
            inner_labs=[baseline.inner_lab()],
            categorical_fis=self.categorical_fi_nicks(),
            survival_fis=self.survival_fi_nicks(),
            generations=[baseline.generations()],
            population=[baseline.population()],
            hof_nick=hof_nick,
            adjuster_regressors=[baseline.adjuster_regressor()],
            covariate_sets=[baseline.covariate_set()],
            categorical_fs_descriptors=[baseline.categorical_fs_descriptor()])) for datasets in self.datasets()]

    def dataset_names(self) -> Sequence[str]:
        return [self.plot_setup().labels_map().apply(d.name()) for d in self.datasets()]
