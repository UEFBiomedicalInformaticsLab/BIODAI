from copy import deepcopy

from hall_of_fame.hall_of_fame import DrippingHallOfFame
from hall_of_fame.hof_utils import iterable_to_hofers
from hall_of_fame.hofers import Hofers
from hall_of_fame.participants import Participants
from individual.fit import Fit


PARETO_NICK = "Pareto"


class Fronts(DrippingHallOfFame):
    __number_of_fronts: int
    __fronts: list[set[Fit]]

    def __init__(self, number_of_fronts: int):
        """With 1 front is the Pareto front. Works also with 0 fronts (no element is kept)."""
        self.__number_of_fronts = number_of_fronts
        self.__fronts = [set() for _ in range(number_of_fronts)]

    @staticmethod
    def __drip_one_front(front: set[Fit], elem: Fit) -> set[Fit]:
        """Returns the elements that have been removed from the front.
        Element is added to the front if needed. It is added as is, remember to deepcopy before calling if needed.
        An element that is different but has the same fitnesses that one already present will be added to the front."""
        elem_fit = elem.get_test_fitness()
        to_remove = set()
        for e in front:
            if e.get_test_fitness().dominates(elem_fit):
                to_remove.add(elem)
                return to_remove
        for e in front:
            if elem_fit.dominates(e.get_test_fitness()):
                to_remove.add(e)
        front.difference_update(to_remove)
        front.add(elem)
        return to_remove

    def add(self, elem: Fit):
        """Proceeds one front at a time, elements not added or removed drip down to the next front."""
        to_drip = set()
        to_drip.add(deepcopy(elem))
        for f in self.__fronts:
            drip_new = set()
            for d in to_drip:
                drip_new.update(self.__drip_one_front(front=f, elem=d))
            to_drip = drip_new

    def hofers(self) -> Hofers:
        participants = Participants()
        for f in self.__fronts:
            participants.update(new_elems=f)
        return iterable_to_hofers(elems=participants.hofers(), name=self.name(), nick=self.nick())

    def name(self) -> str:
        if self.__number_of_fronts == 1:
            return "Pareto front"
        else:
            return str(self.__number_of_fronts) + " fronts"

    def nick(self) -> str:
        if self.__number_of_fronts == 1:
            return PARETO_NICK
        else:
            return str(self.__number_of_fronts) + "fronts"
