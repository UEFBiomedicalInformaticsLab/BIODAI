import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from comparators import Comparator
from individual.peculiar_individual import PeculiarIndividual


class Tournament(ABC):

    @abstractmethod
    def sel_tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:
        raise NotImplementedError()


class TournamentByComparator(Tournament):
    __comparator: Comparator

    def __init__(self, comparator: Comparator):
        self.__comparator = comparator

    def sel_tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:
        """Tournament selection based on comparation between two individuals.

        Generalization written starting from the selTournamentDCD from DEAP.
        The *individuals* sequence length has to
        be a multiple of 4 only if k is equal to the length of individuals.
        Starting from the beginning of the selected individuals, two consecutive
        individuals will be different (assuming all individuals in the input list
        are unique). Each individual from the input list won't be selected more
        than twice.

        Uses module random for random choices.

        :param pop: A list of individuals to select from.
        :param k: The number of individuals to select. Must be less than or equal
                  to len(individuals).
        :returns: A list of selected individuals.
        """

        if k > len(pop):
            raise ValueError("selTournament: k must be less than or equal to individuals length")

        if k == len(pop) and k % 4 != 0:
            raise ValueError("selTournament: k must be divisible by four if k == len(individuals)")

        def tourn(ind1, ind2, comp: Comparator):
            comp_res = comp.compare(ind1, ind2)
            if comp_res < 0:
                return ind1
            elif comp_res > 0:
                return ind2
            else:
                if random.random() < 0.5:
                    return ind1
                else:
                    return ind2

        individuals_1 = random.sample(pop, len(pop))
        individuals_2 = random.sample(pop, len(pop))

        comparator = self.__comparator

        chosen = []
        for i in range(0, k, 4):
            chosen.append(tourn(individuals_1[i], individuals_1[i + 1], comp=comparator))
            chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3], comp=comparator))
            chosen.append(tourn(individuals_2[i], individuals_2[i + 1], comp=comparator))
            chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3], comp=comparator))

        return chosen


class TournamentByPosition(Tournament):
    """Assumes that the individuals are sorted from best to worst and uses just their positions."""

    def sel_tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:

        n_individuals = len(pop)

        if k > n_individuals:
            raise ValueError("selTournament: k must be less than or equal to individuals length")

        if k == n_individuals and k % 4 != 0:
            raise ValueError("selTournament: k must be divisible by four if k == len(individuals)")

        indices_1 = random.sample(range(n_individuals), n_individuals)
        indices_2 = random.sample(range(n_individuals), n_individuals)

        chosen = []
        for i in range(0, k, 4):
            chosen.append(pop[min(indices_1[i], indices_1[i + 1])])
            chosen.append(pop[min(indices_1[i + 2], indices_1[i + 3])])
            chosen.append(pop[min(indices_2[i], indices_2[i + 1])])
            chosen.append(pop[min(indices_2[i + 2], indices_2[i + 3])])

        return chosen


class Elite(Tournament):
    """Assumes that the individuals are sorted from best to worst and uses just their positions.
    Returns the first k individuals."""

    def sel_tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:

        n_individuals = len(pop)

        if k > n_individuals:
            raise ValueError("selTournament: k must be less than or equal to individuals length")

        return pop[:k]


class RandomTournament(Tournament):
    """Returns randomly with repetition."""

    def sel_tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:

        n_individuals = len(pop)

        if k > n_individuals:
            raise ValueError("selTournament: k must be less than or equal to individuals length")

        return random.choices(pop, k=k)
