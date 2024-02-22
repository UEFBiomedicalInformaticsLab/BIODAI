from individual.individual_by_listlike import IndividualByListlike


class DenseIndividual(IndividualByListlike):

    @staticmethod
    def _init_list_like(seq):
        return list(seq)

    def __eq__(self, other):
        return self._list_like() == other
