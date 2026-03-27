from copy import deepcopy
from typing import Iterable

from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from hall_of_fame.hof_utils import iterable_to_hofers
from individual.fit import Fit


class LastPopHof(HallOfFame):
    """Keeps all the solutions (including repetitions) of the last observed population."""
    __elems: list[Fit]

    def __init__(self):
        self.__elems = []

    def hofers(self) -> Hofers:
        return iterable_to_hofers(elems=self.__elems, name=self.name(), nick=self.nick())

    def update(self, new_elems: Iterable[Fit]):
        pass

    def signal_final(self, final_elems: Iterable[Fit]):
        self.__elems = [deepcopy(e) for e in final_elems]
        print("Number of saved elements: " + str(len(self.__elems)))

    def name(self) -> str:
        return "Last population"

    def nick(self) -> str:
        return "last_pop"
