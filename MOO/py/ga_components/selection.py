from abc import ABC, abstractmethod

from individual.peculiar_individual import PeculiarIndividual
from util.named import NickNamed


DEFAULT_N_PARTICIPANTS = 2


class Selection(NickNamed, ABC):

    @abstractmethod
    def select(self, pop: [PeculiarIndividual], pop_size: int) -> [PeculiarIndividual]:
        """Passed population is assumed to be sorted by preference, the first individual is the top one."""
        raise NotImplementedError()


class ElitistSelection(Selection):

    def select(self, pop: [PeculiarIndividual], pop_size: int) -> [PeculiarIndividual]:
        return pop[:pop_size]

    def nick(self) -> str:
        return ""   # No nick since this is the default.

    def name(self) -> str:
        return "elitist"


DEFAULT_SELECTION = ElitistSelection()
DEFAULT_SELECTION_NAME = DEFAULT_SELECTION.name()
