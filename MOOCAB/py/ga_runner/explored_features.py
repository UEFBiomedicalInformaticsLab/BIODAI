from copy import copy

from individual.individual_with_context import IndividualWithContext
from util.sparse_bool_list_by_set import union_of_pair


class ExploredFeatures:
    __union = list[bool]

    def __init__(self):
        self.__union = None

    def update(self, pop):
        for i in pop:
            if self.__union is None:
                self.__union = [False]*len(i)
            if isinstance(i, IndividualWithContext):
                mask = i.active_features_mask()  # We extract the mask for performance reasons.
            else:
                mask = i
            self.__union = union_of_pair(self.__union, mask)

    def num_explored_features(self):
        if self.__union is None:
            return 0
        else:
            return sum(self.__union)

    def explored_features(self):
        return copy(self.__union)

    def __str__(self):
        res = ""
        res += "Number of explored features: " + str(self.num_explored_features()) + "\n"
