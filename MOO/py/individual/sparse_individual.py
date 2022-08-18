from individual.individual_by_listlike import IndividualByListlike
from util.sparse_bool_list_by_set import SparseBoolListBySet, SparseBoolList


class SparseIndividual(IndividualByListlike, SparseBoolList):

    @staticmethod
    def _init_list_like(seq):
        return SparseBoolListBySet(seq=seq)

    def true_positions(self) -> set[int]:
        return self._list_like().true_positions()
