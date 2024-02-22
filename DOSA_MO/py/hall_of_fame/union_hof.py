from collections.abc import Sequence, Iterable
from copy import deepcopy

from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hof_utils import iterable_to_hofers
from hall_of_fame.hofers import Hofers
from hall_of_fame.participants import Participants
from individual.fit import Fit
from util.sequence_utils import sequence_to_string


class UnionHof(HallOfFame):
    """Union of the solutions of a list of HoFs."""
    __inner: Sequence[HallOfFame]

    def __init__(self, inner_hofs: Sequence[HallOfFame]):
        self.__inner = deepcopy(inner_hofs)

    def hofers(self) -> Hofers:
        participants = Participants()
        for hof in self.__inner:
            participants.update(new_elems=hof.hofers())
        return iterable_to_hofers(elems=participants.hofers(), name=self.name(), nick=self.nick())

    def update(self, new_elems: Iterable[Fit]):
        for hof in self.__inner:
            hof.update(new_elems=new_elems)

    def signal_final(self, final_elems: Iterable[Fit]):
        for hof in self.__inner:
            hof.signal_final(final_elems=final_elems)

    def name(self) -> str:
        return "union of " + sequence_to_string(li=[h.name() for h in self.__inner], brackets=False)

    def nick(self) -> str:
        return "union" + sequence_to_string(li=[h.nick() for h in self.__inner], compact=True, brackets=True)
