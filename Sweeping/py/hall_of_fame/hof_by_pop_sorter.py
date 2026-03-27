from copy import deepcopy
from typing import Iterable

from ga_components.sorter.pop_sorter import PopSorter
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.fit import Fit


class HofByPopSorter(HallOfFame):
    """Note that individuals added to this HoF are copied and never updated."""
    __pop_sorter: PopSorter
    __hp_manager: HyperparamManager
    __capacity: int
    __elems: list[Fit]
    __nick: str
    __name: str

    def __init__(self, pop_sorter: PopSorter, capacity: int, hp_manager: HyperparamManager):
        self.__pop_sorter = pop_sorter
        self.__capacity = capacity
        self.__nick = self.__pop_sorter.nick() + str(self.__capacity)
        self.__name = self.__pop_sorter.name() + " top" + str(self.__capacity)
        self.__hp_manager = hp_manager

    def hofers(self) -> Hofers:
        res = []
        for h in self.__elems:
            res.append(deepcopy(h))
        return Hofers(elems=res, name=self.name(), nick=self.nick())

    def update(self, new_elems: Iterable[Fit]):
        self.__elems.extend(new_elems)
        self.__elems = self.__pop_sorter.sort(pop=self.__elems, hp_manager=self.__hp_manager)
        if len(self.__elems) > self.__capacity:
            self.__elems = self.__elems[:self.__capacity]

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name
