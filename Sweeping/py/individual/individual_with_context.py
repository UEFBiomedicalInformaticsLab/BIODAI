from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Iterable, Optional, Sequence

from hyperparam_manager.hyperparam_manager import HyperparamManager
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from individual.confident_individual import ConfidentIndividual
from individual.confident_predictive_individual import ConfidentPredictiveIndividual, ConfidentPredictiveIndividualSV
from individual.fitness.high_best_fitness import HighBestFitness
from individual.fitness.peculiar_fitness import PeculiarFitness
from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from individual.predictive_individual import PredictiveIndividual
from model.sv_model import Predictor
from util.hyperbox.hyperbox import Interval
from util.list_like import BoolListLike
from util.preconditions import check_none
from util.str_utils import iterable_to_string
from util.sparse_bool_list_by_set import SparseBoolList
from util.utils import IllegalStateError


class IndividualWithContext(ConfidentPredictiveIndividual, SparseBoolList, ABC):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified.
    An individual with context has an embedded hp manager that
    allows for direct extraction of the used features. The individual, seen
    as a sequence, is a mask of the used features of the collapsed views.
    I.e. the individual is the result of calling collapsed_used_features_mask of the embedded hp manager on
    the nested individual.
    A hash is defined because it has to be treated as immutable."""
    _individual: ConfidentPredictiveIndividual
    __cached_collapsed_used_features_mask: Optional[BoolListLike]
    __cached_collapsed_used_features_mask_len: Optional[int]
    __hash: Optional[int]

    def __init__(self, individual: ConfidentPredictiveIndividual):
        PredictiveIndividual.__init__(self=self, fitness=None)
        self._individual = check_none(individual)
        self.fitness = None
        if self._individual.has_fitness():
            self.fitness = self._individual.fitness
        self.__cached_collapsed_used_features_mask = None
        self.__cached_collapsed_used_features_mask_len = None
        self.__hash = None

    @abstractmethod
    def _hp_manager(self) -> HyperparamManager:
        raise NotImplementedError()

    def collapsed_used_features_mask(self) -> BoolListLike:
        if self.__cached_collapsed_used_features_mask is None:
            self.__cached_collapsed_used_features_mask = self._hp_manager().collapsed_used_features_mask(
                hyperparams=self._individual)
        return self.__cached_collapsed_used_features_mask

    def __len__(self):
        if self.__cached_collapsed_used_features_mask_len is None:
            self.__cached_collapsed_used_features_mask_len = len(self.collapsed_used_features_mask())
        return self.__cached_collapsed_used_features_mask_len

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, IndividualWithContext):
            return self.collapsed_used_features_mask() == other.collapsed_used_features_mask()
        else:
            return False

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(self._hp_manager().to_tuple(hyperparams=self._individual))
        return self.__hash

    def brief_str(self):
        ret_string = ""
        ret_string += str(self.fitness) + " "
        ret_string += iterable_to_string(self.collapsed_used_features_mask().true_positions())
        return ret_string

    def __str__(self):
        ret_string = ""
        ret_string += str(self.fitness) + " "
        ret_string += str(self.collapsed_used_features_mask())
        return ret_string

    def get_predictors(self) -> Sequence[Predictor]:
        return self._individual.get_predictors()

    def has_fitness(self):
        return self._individual.has_fitness()

    def get_test_fitness(self) -> HighBestFitness:
        return self._individual.get_test_fitness()

    def __getitem__(self, pos):
        return self.collapsed_used_features_mask()[pos]

    def __iter__(self):
        return self.collapsed_used_features_mask().__iter__()

    def sum(self):
        return self.collapsed_used_features_mask().sum()

    def true_positions(self):
        return self.collapsed_used_features_mask().true_positions()

    def to_numpy(self):
        return self.collapsed_used_features_mask().to_numpy()

    def __setitem__(self, key, value):
        raise IllegalStateError()

    def extend(self, iterable: Iterable):
        raise IllegalStateError()

    def append(self, value):
        raise IllegalStateError()

    def modifiable_copy(self) -> PeculiarIndividual:
        res = PeculiarIndividualSparse(n_objectives=self.n_objectives(), seq=self)
        res.fitness = PeculiarFitness.create_from_high_best_fitness(f=self.fitness)
        res.set_predictors(self.get_predictors())
        return res

    def std_dev(self) -> Sequence[Optional[float]]:
        if isinstance(self._individual, ConfidentIndividual):
            return self._individual.std_dev()
        else:
            return [None]*self._individual.n_objectives()

    def ci95(self) -> Sequence[Optional[Interval]]:
        if isinstance(self._individual, ConfidentIndividual):
            return self._individual.ci95()
        else:
            return [None]*self._individual.n_objectives()

    def bootstrap_mean(self) -> Sequence[Optional[float]]:
        if isinstance(self._individual, ConfidentIndividual):
            return self._individual.bootstrap_mean()
        else:
            return [None]*self._individual.n_objectives()


class IndividualWithContextSV(IndividualWithContext, ConfidentPredictiveIndividualSV):
    __hp_manager: SvHyperparamManager

    def __init__(self, individual: ConfidentPredictiveIndividualSV, hp_manager: SvHyperparamManager):
        IndividualWithContext.__init__(self=self, individual=individual)
        self.__hp_manager = check_none(hp_manager)

    def _hp_manager(self) -> SvHyperparamManager:
        return self.__hp_manager
