from abc import ABC, abstractmethod
from typing import Callable, Sequence

from comparators import ComparatorOnDominationAndCrowding
from ga_components.selection import Selection, DEFAULT_SELECTION
from ga_components.sorter.nsga2_with_secondary_comparator import Nsga2WithCrowdingDistance
from ga_components.sorter.sorter_with_clone_index import SorterWithCloneIndex
from ga_components.tournament import Tournament, TournamentByComparator, TournamentByPosition
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual
from ga_components.sorter.pop_sorter import PopSorter
from util.named import NickNamed


class SortingStrategy(NickNamed, ABC):

    @abstractmethod
    def apply_before_selection(
            self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        """Original population is not sorted.
        Individuals may have attributes updated during the process (e.g. crowding distance)."""
        raise NotImplementedError()

    @abstractmethod
    def apply_after_selection(
            self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        """Original population is not sorted.
        Individuals may have attributes updated during the process (e.g. crowding distance)."""
        raise NotImplementedError()

    @abstractmethod
    def tournament(
            self, pop: Sequence[PeculiarIndividual], k) -> Sequence[PeculiarIndividual]:
        raise NotImplementedError()

    @abstractmethod
    def to_be_added_to_stats(self) -> dict[str, Callable[[PeculiarIndividual], float]]:
        raise NotImplementedError()

    @abstractmethod
    def select(self, pop: Sequence[PeculiarIndividual], pop_size: int) -> Sequence[PeculiarIndividual]:
        raise NotImplementedError()

    @abstractmethod
    def basic_algorithm_nick(self) -> str:
        raise NotImplementedError()


class SortingStrategyWithSorter(SortingStrategy):
    __sorter: PopSorter
    __sort_after_selection: bool
    __tournament: Tournament
    __selection: Selection

    def __init__(self, sorter: PopSorter, tournament: Tournament, sort_after_selection: bool,
                 selection: Selection = DEFAULT_SELECTION):
        self.__sorter = sorter
        self.__tournament = tournament
        self.__sort_after_selection = sort_after_selection
        self.__selection = selection

    def apply_before_selection(
            self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        return self.__sorter.sort(pop=pop, hp_manager=hp_manager)

    def select(self, pop: Sequence[PeculiarIndividual], pop_size: int) -> Sequence[PeculiarIndividual]:
        return self.__selection.select(pop=pop, pop_size=pop_size)

    def apply_after_selection(
            self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        if self.__sort_after_selection:
            return self.__sorter.sort(pop=pop, hp_manager=hp_manager)
        else:
            return pop

    def tournament(self, pop: Sequence[PeculiarIndividual], k) -> Sequence[PeculiarIndividual]:
        return self.__tournament.sel_tournament(pop=pop, k=k)

    def to_be_added_to_stats(self) -> dict[str, Callable[[PeculiarIndividual], float]]:
        return self.__sorter.to_be_added_to_stats()

    def nick(self) -> str:
        res = self.__sorter.nick()
        if self.__sort_after_selection:
            res += "Full"
        if self.__selection.nick() != "":
            res += "_" + self.__selection.nick()
        return res

    def sorter_nick(self) -> str:
        return self.__sorter.nick()

    def selection_nick(self) -> str:
        return self.__selection.nick()

    def name(self) -> str:
        res = self.__sorter.name()
        if self.__sort_after_selection:
            res += " full"
        if self.__selection.name() != "":
            res += " " + self.__selection.name()
        return res

    def __str__(self) -> str:
        res = str(self.__sorter)
        if self.__sort_after_selection:
            res += " full"
        res += " " + str(self.__selection)
        return res

    def basic_algorithm_nick(self) -> str:
        return self.__sorter.basic_algorithm_nick()

    def _sorter(self) -> PopSorter:
        return self.__sorter

    def _selection(self) -> Selection:
        return self.__selection


class SortingStrategyCrowd(SortingStrategyWithSorter):

    def __init__(self, selection: Selection = DEFAULT_SELECTION):
        SortingStrategyWithSorter.__init__(
            self=self,
            sorter=Nsga2WithCrowdingDistance(),
            tournament=TournamentByComparator(comparator=ComparatorOnDominationAndCrowding()),
            sort_after_selection=False,
            selection=selection)


class SortingStrategyCrowdFull(SortingStrategyWithSorter):

    def __init__(self, selection: Selection = DEFAULT_SELECTION):
        SortingStrategyWithSorter.__init__(
            self=self,
            sorter=Nsga2WithCrowdingDistance(),
            tournament=TournamentByPosition(),
            sort_after_selection=True, selection=selection)


class SortingStrategyCrowdCI(SortingStrategyWithSorter):

    def __init__(self, selection: Selection = DEFAULT_SELECTION):
        SortingStrategyWithSorter.__init__(
            self=self,
            sorter=SorterWithCloneIndex(inner_sorter=Nsga2WithCrowdingDistance()),
            tournament=TournamentByPosition(),
            sort_after_selection=True, selection=selection)

    def nick(self) -> str:
        res = self.sorter_nick()
        if self.selection_nick() != "":
            res += "_" + self.selection_nick()
        return res
