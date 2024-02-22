from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Optional, Union

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from load_omics_views import MRNA_NAME
from objective.objective_computer import ObjectiveComputer
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabelByComputers, ObjectivesDirFromLabel
from plots.hofs_plotter.plot_setup import PlotSetup, PlotSetupWithDefaultLabels
from plots.plot_labels import ALL_INNER_LABS, ALL_MAIN_LABS, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF
from util.named import NickNamed


class TestBattery(NickNamed, ABC):
    __objective_computers: Sequence[ObjectiveComputer]
    __views: Sequence[str]
    __main_labs: Sequence[str]
    __inner_labs: Sequence[str]
    __cox_fi: bool
    __nick: str
    __plot_setup: PlotSetup
    __generations: Optional[GenerationsStrategy]
    __adjuster_regressors: Sequence[Union[None, str]]

    def __init__(self,
                 objective_computers: Sequence[ObjectiveComputer],
                 views: Sequence[str] = (MRNA_NAME,),
                 main_labs: Sequence[str] = ALL_MAIN_LABS,
                 generations: Optional[GenerationsStrategy] = None,
                 inner_labs: Sequence[str] = ALL_INNER_LABS,
                 cox_fi: bool = True,
                 adjuster_regressors: Sequence[Optional[str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
                 nick: Optional[str] = None,
                 plot_setup: PlotSetup = PlotSetupWithDefaultLabels()):
        self.__objective_computers = objective_computers
        self.__views = views
        self.__main_labs = main_labs
        self.__generations = generations
        self.__inner_labs = inner_labs
        self.__cox_fi = cox_fi
        self.__plot_setup = plot_setup
        self.__adjuster_regressors = adjuster_regressors
        if nick is None:
            self.__nick = self._automatic_nick()
        else:
            self.__nick = nick

    def objective_computers(self) -> Sequence[ObjectiveComputer]:
        return self.__objective_computers

    def dir_from_label(self) -> ObjectivesDirFromLabel:
        return ObjectivesDirFromLabelByComputers(objectives=self.objective_computers())

    def n_objectives(self) -> int:
        return len(self.objective_computers())

    def views(self) -> Sequence[str]:
        return self.__views

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

    def cox_fi(self) -> bool:
        return self.__cox_fi

    def plot_setup(self) -> PlotSetup:
        return self.__plot_setup

    def generations(self) -> Optional[GenerationsStrategy]:
        return self.__generations

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

    def type_str(self) -> str:
        if self.is_external():
            return "external"
        else:
            return "cv"
