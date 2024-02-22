from comparators import ComparatorOnSum
from hall_of_fame.hall_of_fame_by_comparator import HallOfFameByComparator, PruneAtSize


class HofBySum(HallOfFameByComparator):
    __name: str
    __nick: str

    def __init__(self, size: int):
        HallOfFameByComparator.__init__(self, comparator=ComparatorOnSum(), pruner=PruneAtSize(size=size))
        self.__name = "best sums with size " + str(size)
        self.__nick = "sum" + str(size)

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick
