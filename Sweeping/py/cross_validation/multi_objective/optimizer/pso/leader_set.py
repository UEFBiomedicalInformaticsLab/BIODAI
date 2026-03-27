import random
from collections.abc import Iterable
from operator import attrgetter

from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist

from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_dense import PeculiarIndividualDense


class LeaderSet:
    """Does not keep clones. """
    __capacity: int
    __leaders: list[PeculiarIndividual]

    def __init__(self, capacity: int):
        self.__capacity = capacity
        self.__leaders = []

    def __sort_leaders(self):
        assignCrowdingDist(self.__leaders)
        self.__leaders = sorted(self.__leaders, key=attrgetter("fitness.crowding_dist"), reverse=True)

    def update(self, new_elems: Iterable[PeculiarIndividual]):
        leaders_set = set(self.__leaders)
        copies = []
        for e in new_elems:
            c = PeculiarIndividualDense(n_objectives=e.n_objectives(), seq=e)
            c.fitness = e.get_test_fitness()
            copies.append(c)
        leaders_set.update(copies)
        # We have to safe copy because crowding distance will be specific for leader set.
        self.__leaders = sortNondominated(individuals=leaders_set, first_front_only=True, k=len(leaders_set))[0]
        self.__sort_leaders()
        if len(self.__leaders) > self.__capacity:
            self.__leaders = self.__leaders[:self.__capacity]
            # After removing elements crowding distance has to be updated.
            self.__sort_leaders()

    def tournament_select(self) -> PeculiarIndividual:
        samples = random.choices(range(len(self.__leaders)), k=2)
        selected = min(samples)
        return self.__leaders[selected]
