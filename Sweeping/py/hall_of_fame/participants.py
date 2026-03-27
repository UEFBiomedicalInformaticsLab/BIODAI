from copy import deepcopy

from hall_of_fame.hall_of_fame import DrippingHallOfFame
from hall_of_fame.hof_utils import iterable_to_hofers
from hall_of_fame.hofers import Hofers
from individual.fit import Fit


class Participants(DrippingHallOfFame):
    """All unique elements are kept."""
    __elems: set[Fit]

    def __init__(self):
        self.__elems = set()

    def add(self, elem):
        self.__elems.add(deepcopy(elem))

    def hofers(self) -> Hofers:
        return iterable_to_hofers(elems=self.__elems, name=self.name(), nick=self.nick())

    def nick(self) -> str:
        return "participants"

    def name(self) -> str:
        return self.nick()
