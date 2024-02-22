from collections.abc import Sequence
from typing import Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from load_omics_views import MRNA_NAME
from objective.objective_computer import ObjectiveComputer
from plots.archives.archives_utils import dataset_base_dir
from plots.archives.automated_hofs_archive import all_hof_combinations_external, existing_hofs
from plots.archives.test_battery import TestBattery
from plots.hofs_plotter.plot_setup import PlotSetup, PlotSetupWithDefaultLabels
from plots.plot_labels import ALL_MAIN_LABS, ALL_INNER_LABS, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF
from util.sequence_utils import sequence_to_string, clean_redundant_subsequences


class TestBatteryExternal(TestBattery):
    __internal_dataset_label: str
    __external_dataset_label: str

    def __init__(self,
                 objective_computers: Sequence[ObjectiveComputer],
                 internal_dataset_label: str,
                 external_dataset_label: str,
                 views: Sequence[str] = (MRNA_NAME,),
                 main_labs: Sequence[str] = ALL_MAIN_LABS,
                 generations: Optional[GenerationsStrategy] = None,
                 inner_labs: Sequence[str] = ALL_INNER_LABS,
                 cox_fi: bool = True,
                 adjuster_regressors: Sequence[Optional[str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
                 nick: Optional[str] = None,
                 plot_setup: PlotSetup = PlotSetupWithDefaultLabels()):
        self.__internal_dataset_label = internal_dataset_label
        self.__external_dataset_label = external_dataset_label
        TestBattery.__init__(self,
                             objective_computers=objective_computers,
                             views=views,
                             main_labs=main_labs,
                             generations=generations,
                             inner_labs=inner_labs,
                             cox_fi=cox_fi,
                             adjuster_regressors=adjuster_regressors,
                             nick=nick,
                             plot_setup=plot_setup)

    def internal_dataset_label(self) -> str:
        return self.__internal_dataset_label

    def external_dataset_label(self) -> str:
        return self.__external_dataset_label

    def external_dataset_nick(self) -> str:
        return dataset_base_dir(dataset_lab=self.external_dataset_label())

    def _automatic_nick(self) -> str:
        return self.__internal_dataset_label + "_" + self.__external_dataset_label

    def is_external(self) -> bool:
        return True

    def existing_flat_hofs(self, hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
        return existing_hofs(all_hof_combinations_external(
            dataset_lab=self.internal_dataset_label(),
            external_nick=self.external_dataset_nick(),
            main_labs=self.main_labs(),
            inner_labs=self.inner_labs(),
            dir_from_label=self.dir_from_label(),
            views=self.views(),
            cox_fi=self.cox_fi(),
            generations=self.generations(),
            hof_nick=hof_nick,
            adjuster_regressors=self.adjuster_regressors()))

    def dataset_report_path_part(self) -> str:
        battery_nick = self.nick()
        objective_computers = self.objective_computers()
        if battery_nick is None:
            dataset_path_part = self.internal_dataset_label() + "_" + self.external_dataset_label()
        else:
            dataset_path_part = battery_nick
        objectives_str = sequence_to_string(sorted([o.nick() for o in objective_computers]),
                                            compact=True, separator="_", brackets=False)
        return dataset_path_part + "/" + objectives_str

    def existing_nested_hofs(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """A list of sequences, one for each inner model."""
        dir_from_label = self.dir_from_label()
        res = []
        for inner in self.inner_labs():
            hofs = all_hof_combinations_external(
                main_labs=self.main_labs(),
                views=self.views(),
                dir_from_label=dir_from_label,
                dataset_lab=self.internal_dataset_label(),
                external_nick=self.external_dataset_nick(),
                inner_labs=[inner],
                cox_fi=self.cox_fi(),
                generations=self.generations(),
                hof_nick=hof_nick,
                adjuster_regressors=self.adjuster_regressors())
            existing = existing_hofs(hofs)
            if len(existing) > 0:
                res.append(existing)
        return clean_redundant_subsequences(res)

    def existing_hofs_grouped_by_dataset_and_inner(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        return self.existing_nested_hofs(hof_nick=hof_nick)

    def flat_hofs(self, hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
        """Sequences of saved hofs."""
        dir_from_label = self.dir_from_label()
        return all_hof_combinations_external(
            dataset_lab=self.internal_dataset_label(),
            external_nick=self.external_dataset_nick(),
            main_labs=self.main_labs(),
            views=self.views(),
            dir_from_label=dir_from_label,
            inner_labs=self.inner_labs(),
            cox_fi=self.cox_fi(),
            generations=self.generations(),
            hof_nick=hof_nick,
            adjuster_regressors=self.adjuster_regressors())
