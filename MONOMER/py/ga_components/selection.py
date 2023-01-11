from abc import ABC, abstractmethod

from ga_components.tournament import ExtractionByTournament
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


class TournamentExtraction(Selection):
    __extractor: ExtractionByTournament

    def __init__(self, n_participants: int = DEFAULT_N_PARTICIPANTS):
        self.__extractor = ExtractionByTournament(n_participants=n_participants)

    def select(self, pop: [PeculiarIndividual], pop_size: int) -> [PeculiarIndividual]:
        return self.__extractor.sel_tournament(pop=pop, k=pop_size)

    @staticmethod
    def base_nick() -> str:
        return "tourn"

    def nick(self) -> str:
        return self.base_nick() + str(self.__extractor.n_participants())

    def name(self) -> str:
        return "tournament(" + str(self.__extractor.n_participants()) + ")"


DEFAULT_SELECTION = ElitistSelection()
DEFAULT_SELECTION_NAME = DEFAULT_SELECTION.name()
