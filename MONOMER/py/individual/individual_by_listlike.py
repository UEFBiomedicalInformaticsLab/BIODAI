from abc import abstractmethod

from sortedcontainers import SortedSet

from individual.Individual import Individual
from util.list_like import ListLike


class IndividualByListlike(Individual):
    __list_like: ListLike

    def __init__(self, seq=()):
        self.__list_like = self._init_list_like(seq)

    @staticmethod
    @abstractmethod
    def _init_list_like(seq) -> ListLike:
        raise NotImplementedError()

    def _list_like(self):
        return self.__list_like

    def has_fitness(self):
        return False

    def __len__(self):
        return len(self.__list_like)

    def __getitem__(self, pos):
        return self.__list_like.__getitem__(pos)

    def __setitem__(self, pos, data):
        self.__list_like[pos] = data

    def extend(self, lst: list):
        self.__list_like.extend(lst)

    def __iter__(self):
        return self.__list_like.__iter__()

    def to_numpy(self):
        return self.__list_like.to_numpy()

    def sum(self):
        return self.__list_like.sum()

    def append(self, value):
        """Appends just one element."""
        self.__list_like.append(value)

    def __str__(self):
        ret_string = "Individual object with chromosome:\n"
        ret_string += str(self.__list_like) + "\n"
        return ret_string

    def __eq__(self, other):
        if isinstance(other, IndividualByListlike):
            return self.__list_like == other.__list_like
        else:
            return False

    def __hash__(self):
        return hash(self.__list_like)

    def true_positions(self) -> SortedSet[int]:
        return self.__list_like.true_positions()
