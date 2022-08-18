from typing import Iterable

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.predictive_individual import PredictiveIndividual
from model.model import Classifier
from util.list_like import ListLike
from util.preconditions import check_none
from util.sequence_utils import sequence_to_string
from util.sparse_bool_list_by_set import SparseBoolList
from util.utils import IllegalStateError


class IndividualWithContext(PredictiveIndividual, SparseBoolList):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified."""
    __individual: PredictiveIndividual
    __hp_manager: HyperparamManager
    __cached_active_features_mask: ListLike
    __len: int

    def __init__(self, individual: PredictiveIndividual, hp_manager: HyperparamManager):
        self.__individual = check_none(individual)
        self.__hp_manager = check_none(hp_manager)
        self.fitness = None
        if self.__individual.has_fitness():
            self.fitness = self.__individual.fitness
        self.__cached_active_features_mask = None
        self.__len = hp_manager.active_features_mask_len(hyperparams=individual)

    def active_features_mask(self):
        if self.__cached_active_features_mask is None:
            self.__cached_active_features_mask = self.__hp_manager.active_features_mask(hyperparams=self.__individual)
        return self.__cached_active_features_mask

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, IndividualWithContext):
            return self.active_features_mask() == other.active_features_mask()
        else:
            return False

    def __hash__(self):
        return hash(self.active_features_mask())

    def brief_str(self):
        ret_string = ""
        ret_string += str(self.fitness) + " "
        ret_string += sequence_to_string(self.active_features_mask().true_positions())
        return ret_string

    def __str__(self):
        ret_string = ""
        ret_string += str(self.fitness) + " "
        ret_string += str(self.active_features_mask())
        return ret_string

    def get_predictors(self) -> [Classifier]:
        return self.__individual.get_predictors()

    def has_fitness(self):
        return self.__individual.has_fitness()

    def __len__(self):
        return self.__len

    def __getitem__(self, pos):
        return self.active_features_mask()[pos]

    def __iter__(self):
        return self.active_features_mask().__iter__()

    def sum(self):
        return self.active_features_mask().sum()

    def true_positions(self):
        return self.active_features_mask().true_positions()

    def to_numpy(self):
        return self.active_features_mask().to_numpy()

    def __setitem__(self, key, value):
        raise IllegalStateError()

    def extend(self, iterable: Iterable):
        raise IllegalStateError()

    def append(self, value):
        raise IllegalStateError()
