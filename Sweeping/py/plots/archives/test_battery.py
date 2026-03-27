from abc import abstractmethod, ABC
from collections.abc import Sequence, Iterable
from typing import Optional, Union

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.hof_names import PARETO_NICK
from location_manager.location_manager_utils import COVARIATES_DEFAULT, DEFAULT_CATEGORICAL_FIS, DEFAULT_SURVIVAL_FIS, \
    VIEWS_DEFAULT
from objective.objective_computer import ObjectiveComputer
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabelByComputers, ObjectivesDirFromLabel
from plots.archives.optimizer_descriptor import OptimizerDescriptor
from plots.hofs_plotter.plot_setup import PlotSetup, PlotSetupWithDefaultLabels
from plots.plot_labels import ALL_INNER_LABS, ALL_MAIN_LABS, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF
from univariate_feature_selection.univariate_feature_selector_descriptor import ANOVA_NICK, LOGISTIC_FDR_SELECTOR_NICK, \
    AnovaCategoricalDescriptor, ManyFeatureSelectorClassDescriptor, \
    FdrManyFeatureSelectorClassDescriptor
from univariate_property_computer.univariate_property_computer_descriptor import LogUnivariatePvalComputerDescriptor
from util.named import NickNamed
from util.printer.printer import OutPrinter
from views.adjusted_view_definition import AdjustedViewDef

DEFAULT_VIEW_DEFS: Sequence[AdjustedViewDef] = (VIEWS_DEFAULT,)
DEFAULT_COVARIATE_SETS = (COVARIATES_DEFAULT,)
DEFAULT_CATEGORICAL_FS_NICKS = (ANOVA_NICK, LOGISTIC_FDR_SELECTOR_NICK)
DEFAULT_FDR_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_CATEGORICAL_FS_DESCRIPTORS = tuple(
        [AnovaCategoricalDescriptor()] +
        [FdrManyFeatureSelectorClassDescriptor(
            computer=LogUnivariatePvalComputerDescriptor(), fdr_threshold=t) for t in DEFAULT_FDR_THRESHOLDS])



class TestBattery(NickNamed, ABC):
    __objective_computers: Sequence[ObjectiveComputer]
    __view_defs: Sequence[AdjustedViewDef]
    __main_labs: Sequence[str]
    __inner_labs: Sequence[str]
    __categorical_fi_nicks: Sequence[str]
    __survival_fi_nicks: Sequence[str]
    __nick: str
    __plot_setup: PlotSetup
    __generations: Optional[Sequence[GenerationsStrategy]]
    __population: Optional[Sequence[int]]
    __adjuster_regressors: Sequence[Optional[str]]
    __baseline: Optional[OptimizerDescriptor]
    __covariate_sets: Sequence[set[str]]
    __categorical_fs_descriptors: Iterable[ManyFeatureSelectorClassDescriptor]

    def __init__(self,
                 objective_computers: Sequence[ObjectiveComputer],
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
                     ManyFeatureSelectorClassDescriptor] = DEFAULT_CATEGORICAL_FS_DESCRIPTORS):
        """Views is a sequence where each element is a combination of views.
        Baseline must be included in the battery."""
        self.__objective_computers = objective_computers
        self.__view_defs = view_defs
        self.__main_labs = main_labs
        self.__generations = generations
        self.__population = population
        self.__inner_labs = inner_labs
        self.__categorical_fi_nicks = categorical_fi_nicks
        self.__survival_fi_nicks = survival_fi_nicks
        self.__plot_setup = plot_setup
        self.__adjuster_regressors = adjuster_regressors
        self.__covariate_sets = covariate_sets
        self.__categorical_fs_descriptors = categorical_fs_descriptors
        if nick is None:
            self.__nick = self._automatic_nick()
        else:
            self.__nick = nick
        if baseline is not None:
            if not baseline.is_included(
                    main_labs=main_labs, inner_labs=inner_labs, generations=generations, population=population,
                    categorical_fi_nicks=categorical_fi_nicks, survival_fi_nicks=survival_fi_nicks,
                    adjuster_regressors=adjuster_regressors, view_defs=view_defs,
                    covariate_sets=covariate_sets, categorical_fs_descriptors=self.__categorical_fs_descriptors,
                    printer=OutPrinter()):
                raise ValueError("Baseline not included in battery " + self.__nick)
        self.__baseline = baseline


    def objective_computers(self) -> Sequence[ObjectiveComputer]:
        return self.__objective_computers

    def dir_from_label(self) -> ObjectivesDirFromLabel:
        return ObjectivesDirFromLabelByComputers(objectives=self.objective_computers())

    def n_objectives(self) -> int:
        return len(self.objective_computers())

    def view_defs(self) -> Sequence[AdjustedViewDef]:
        return self.__view_defs

    def covariate_sets(self) -> Sequence[set[str]]:
        return self.__covariate_sets

    def main_labs(self) -> Sequence[str]:
        return self.__main_labs

    def inner_labs(self) -> Sequence[str]:
        return self.__inner_labs

    def n_inner_labs(self) -> int:
        return len(self.inner_labs())

    @abstractmethod
    def _automatic_nick(self) -> str:
        raise NotImplementedError()

    def nick(self) -> str:
        return self.__nick

    def categorical_fi_nicks(self) -> Sequence[str]:
        return self.__categorical_fi_nicks

    def survival_fi_nicks(self) -> Sequence[str]:
        return self.__survival_fi_nicks

    def plot_setup(self) -> PlotSetup:
        return self.__plot_setup

    def generations(self) -> Optional[Sequence[GenerationsStrategy]]:
        return self.__generations

    def population(self) -> Optional[Sequence[int]]:
        return self.__population

    def adjuster_regressors(self) -> Sequence[Union[None, str]]:
        return self.__adjuster_regressors

    @abstractmethod
    def is_external(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def existing_hofs_grouped_by_dataset_and_inner(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """Returns an outer list element for each included combination dataset-inner model.
        These list elements are sequences of saved hofs.
        The datasets are in the same order of method dataset_labels."""
        raise NotImplementedError()

    @abstractmethod
    def existing_flat_hofs(self, hof_nick: str = PARETO_NICK) -> list[Sequence[SavedHoF]]:
        """Returns a list element for each included dataset. List elements are sequences of saved hofs."""
        raise NotImplementedError()

    @abstractmethod
    def existing_hofs_grouped_by_inner_and_dataset(self, hof_nick: str = PARETO_NICK
                                                   ) -> list[Sequence[Sequence[SavedHoF]]]:
        """Returns an outer list element for each included combination inner-dataset model.
        These list elements are sequences of saved hofs.
        The datasets are in the same order of method datasets/dataset_labels."""
        raise NotImplementedError()

    def type_str(self) -> str:
        if self.is_external():
            return "external"
        else:
            return "cv"

    def baseline(self) -> Optional[OptimizerDescriptor]:
        return self.__baseline

    @abstractmethod
    def baseline_hofs(self, hof_nick: str = PARETO_NICK) -> Sequence[Sequence[SavedHoF]]:
        """One sequence for each dataset. The datasets are in the same order of method datasets/dataset_labels.
        Sequence can be empty if no suitable hof is found.
        There might be more than one element if there are
        multiple seeds and/or location managers."""
        raise NotImplementedError()

    @abstractmethod
    def dataset_names(self) -> Sequence[str]:
        """One name for each dataset or for each pair internal - external in case of external validation."""
        raise NotImplementedError()

    def categorical_fs_descriptors(self) -> Iterable[ManyFeatureSelectorClassDescriptor]:
        return self.__categorical_fs_descriptors
