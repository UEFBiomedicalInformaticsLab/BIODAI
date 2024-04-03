from typing import Sequence, Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from objective.objective_computer import ObjectiveComputer
from plots.archives.automated_hofs_archive import all_hof_combinations_cv
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabelByComputers
from plots.archives.test_battery import TestBattery, DEFAULT_VIEW_SETS
from plots.hofs_plotter.plot_setup import PlotSetup, PlotSetupWithDefaultLabels
from plots.plot_labels import ALL_MAIN_LABS, ALL_INNER_LABS
from plots.saved_hof import SavedHoF
from util.sequence_utils import clean_redundant_subsequences, sequence_to_string


DEFAULT_ADJUSTER_REGRESSORS_LABS = (None,)


class TestBatteryCV(TestBattery):
    __n_outer_folds: int
    __dataset_labels: Sequence[str]
    __cv_repeats: int

    def __init__(self,
                 objective_computers: Sequence[ObjectiveComputer],
                 n_outer_folds: int,
                 dataset_labels: Sequence[str],
                 view_sets: Sequence[set[str]] = DEFAULT_VIEW_SETS,
                 main_labs: Sequence[str] = ALL_MAIN_LABS,
                 cv_repeats: int = 1,
                 generations: Optional[GenerationsStrategy] = None,
                 population: Optional[int] = None,
                 inner_labs: Sequence[str] = ALL_INNER_LABS,
                 cox_fi: bool = True,
                 adjuster_regressors: Sequence[Optional[str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
                 nick: Optional[str] = None,
                 plot_setup: PlotSetup = PlotSetupWithDefaultLabels()):
        self.__n_outer_folds = n_outer_folds
        self.__dataset_labels = dataset_labels
        self.__cv_repeats = cv_repeats
        TestBattery.__init__(self,
                             objective_computers=objective_computers,
                             view_sets=view_sets,
                             main_labs=main_labs,
                             generations=generations,
                             population=population,
                             inner_labs=inner_labs,
                             cox_fi=cox_fi,
                             adjuster_regressors=adjuster_regressors,
                             nick=nick,
                             plot_setup=plot_setup)

    def n_outer_folds(self) -> int:
        return self.__n_outer_folds

    def dataset_labels(self) -> Sequence[str]:
        return self.__dataset_labels

    def n_datasets(self) -> int:
        return len(self.dataset_labels())

    def cv_repeats(self) -> int:
        return self.__cv_repeats

    def flat_hofs_for_dataset(self, dataset_lab: str, hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
        """Sequences of saved hofs for the selected dataset."""
        dir_from_label = ObjectivesDirFromLabelByComputers(objectives=self.objective_computers())
        return all_hof_combinations_cv(
                main_labs=self.main_labs(),
                n_outer_folds=self.n_outer_folds(),
                cv_repeats=self.cv_repeats(),
                view_sets=self.view_sets(),
                dir_from_label=dir_from_label,
                dataset_lab=dataset_lab,
                inner_labs=self.inner_labs(),
                cox_fi=self.cox_fi(),
                generations=self.generations(),
                population=self.population(),
                hof_nick=hof_nick,
                adjuster_regressors=self.adjuster_regressors())

    def existing_flat_hofs_for_dataset(self, dataset_lab: str, hof_nick: str = PARETO_NICK) -> Sequence[SavedHoF]:
        """Sequences of saved hofs for the selected dataset."""
        res = []
        for hof in self.flat_hofs_for_dataset(dataset_lab=dataset_lab, hof_nick=hof_nick):
            if hof.path_exists():
                res.append(hof)
        return res

    def existing_nested_hofs_for_dataset(
            self, dataset_lab: str, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """A list of sequences, one for each inner model."""
        dir_from_label = ObjectivesDirFromLabelByComputers(objectives=self.objective_computers())
        res = []
        for inner in self.inner_labs():
            hofs = all_hof_combinations_cv(
                main_labs=self.main_labs(),
                n_outer_folds=self.n_outer_folds(),
                cv_repeats=self.cv_repeats(),
                view_sets=self.view_sets(),
                dir_from_label=dir_from_label,
                dataset_lab=dataset_lab,
                inner_labs=[inner],
                cox_fi=self.cox_fi(),
                generations=self.generations(),
                population=self.population(),
                hof_nick=hof_nick,
                adjuster_regressors=self.adjuster_regressors())
            existing_hofs = []
            for h in hofs:
                if h.path_exists():
                    existing_hofs.append(h)
            if len(existing_hofs) > 0:
                res.append(existing_hofs)
        res = clean_redundant_subsequences(res)
        return res

    def flat_hofs(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """Returns a list element for each included dataset. List elements are sequences of saved hofs.
        The datasets are in the same order of method dataset_labels"""
        return [self.flat_hofs_for_dataset(
            dataset_lab=dataset_lab, hof_nick=hof_nick) for dataset_lab in self.dataset_labels()]

    def existing_nested_hofs(self, hof_nick: str = PARETO_NICK) -> list[list[Sequence[SavedHoF]]]:
        """Returns an outer list element for each included dataset. For each dataset there is a list with an
         element for each inner model, these list elements are sequences of saved hofs.
        The datasets are in the same order of method dataset_labels"""
        return [self.existing_nested_hofs_for_dataset(
            dataset_lab=dataset_lab, hof_nick=hof_nick) for dataset_lab in self.dataset_labels()]

    def existing_hofs_grouped_by_dataset_and_inner(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """Returns an outer list element for each included combination dataset-inner model.
        These list elements are sequences of saved hofs.
        The datasets are in the same order of method dataset_labels."""
        res = []
        for dataset_lab in self.dataset_labels():
            res.extend(self.existing_nested_hofs_for_dataset(dataset_lab=dataset_lab, hof_nick=hof_nick))
        return res

    def __automatic_nick(self) -> str:
        res = ""
        for lab in self.dataset_labels():
            if res != "":
                res += "_"
            res += lab
        return res

    def dataset_report_path_part(self, dataset_lab: str) -> str:
        battery_nick = self.nick()
        objective_computers = self.objective_computers()
        single_dataset = self.n_datasets() < 2
        if battery_nick is None:
            dataset_path_part = dataset_lab
        else:
            if single_dataset:
                dataset_path_part = battery_nick
            else:
                dataset_path_part = battery_nick + "_" + dataset_lab
        objectives_str = sequence_to_string(sorted([o.nick() for o in objective_computers]),
                                            compact=True, separator="_", brackets=False)
        return dataset_path_part + "/" + objectives_str

    def optimizer_directories(self) -> list[str]:
        res = set()
        hofs = self.flat_hofs()
        for dataset_hofs in hofs:
            for h in dataset_hofs:
                res.add(h.optimizer_dir())
        return list(res)

    def _automatic_nick(self) -> str:
        res = ""
        for lab in self.dataset_labels():
            if res != "":
                res += "_"
            res += lab
        return res

    def is_external(self) -> bool:
        return False
