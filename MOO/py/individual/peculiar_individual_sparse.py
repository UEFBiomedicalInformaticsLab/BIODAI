from copy import deepcopy

from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from individual.sparse_individual import SparseIndividual


class PeculiarIndividualSparse(PeculiarIndividualByListlike, SparseIndividual):

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        res = PeculiarIndividualSparse(n_objectives=self.n_objectives(), seq=self)
        res.fitness = deepcopy(self.fitness)
        res.set_stats(self.get_stats())
        res.set_predictors(self.get_predictors())
        res.set_crowding_distance(self.get_crowding_distance())
        res.set_peculiarity(self.get_peculiarity())
        res.set_social_space(self.get_social_space())
        return res
