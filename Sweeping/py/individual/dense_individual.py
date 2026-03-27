from individual.individual_by_listlike import IndividualByListlike


class DenseIndividual(IndividualByListlike):

    @staticmethod
    def _init_list_like(seq):
        """The sequence is safe-copied."""
        return list(seq)

    def __eq__(self, other):
        return self._list_like() == other

    def __hash__(self):
        return hash(tuple(self._list_like()))
