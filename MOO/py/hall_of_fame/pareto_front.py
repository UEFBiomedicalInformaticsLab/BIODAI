from copy import deepcopy

from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from individual.fit import Fit


PARETO_NICK = "Pareto"


class ParetoFront(HallOfFame):
    __elems: set[Fit]

    def __init__(self):
        self.__elems = set()

    def add(self, elem: Fit):
        elem_fit = elem.get_fitness()
        for e in self.__elems:
            if e.get_fitness().dominates(elem_fit):
                return
        to_remove = set()
        for e in self.__elems:
            if elem_fit.dominates(e.get_fitness()):
                to_remove.add(e)
        self.__elems.difference_update(to_remove)
        self.__elems.add(deepcopy(elem))

    def hofers(self) -> Hofers:
        res = []
        for h in self.__elems:
            res.append(deepcopy(h))
        res.sort(key=lambda e: e.get_fitness(), reverse=True)
        return Hofers(elems=res, name=self.name(), nick=self.nick())

    def name(self) -> str:
        return "Pareto front"

    def nick(self) -> str:
        return PARETO_NICK
