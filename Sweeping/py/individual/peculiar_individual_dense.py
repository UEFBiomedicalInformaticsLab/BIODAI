from individual.dense_individual import DenseIndividual
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike


class PeculiarIndividualDense(PeculiarIndividualByListlike, DenseIndividual):

    def __eq__(self, other):
        return DenseIndividual.__eq__(self, other)

    def __hash__(self):
        return DenseIndividual.__hash__(self)
