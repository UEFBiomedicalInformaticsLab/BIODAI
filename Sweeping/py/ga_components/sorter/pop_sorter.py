from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual
from util.named import NickNamed


class PopSorter(NickNamed, ABC):

    @abstractmethod
    def sort(self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        """Original population is not sorted.
        Individuals may have attributes updated during the process (e.g. crowding distance)."""
        raise NotImplementedError()

    def to_be_added_to_stats(self) -> dict[str, Callable[[PeculiarIndividual], float]]:
        """Override to add to stats."""
        return {}

    @abstractmethod
    def basic_algorithm_nick(self) -> str:
        raise NotImplementedError()
