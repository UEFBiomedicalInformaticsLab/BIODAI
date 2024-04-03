from abc import ABC

from util.list_like import ListLike


class Individual(ListLike, ABC):

    def has_fitness(self):
        return False
