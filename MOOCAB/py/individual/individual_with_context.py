from __future__ import annotations

from copy import copy, deepcopy
from typing import Iterable, Optional, Sequence

from hyperparam_manager.dummy_hp_manager import DummyHpManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.confident_individual import ConfidentIndividual
from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from individual.predictive_individual import PredictiveIndividual
from model.model import Classifier
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.hyperbox.hyperbox import Interval
from util.list_like import ListLike
from util.preconditions import check_none
from util.sequence_utils import sequence_to_string
from util.sparse_bool_list_by_set import SparseBoolList
from util.utils import IllegalStateError


class IndividualWithContext(PredictiveIndividual, ConfidentIndividual, SparseBoolList):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified."""
    _individual: PredictiveIndividual  # TODO Can be a more specific type with confidence.
    __hp_manager: HyperparamManager
    __cached_active_features_mask: Optional[ListLike]
    __len: int

    def __init__(self, individual: PredictiveIndividual, hp_manager: HyperparamManager):
        PredictiveIndividual.__init__(self=self, fitness=None)
        self._individual = check_none(individual)
        self.__hp_manager = check_none(hp_manager)
        self.fitness = None
        if self._individual.has_fitness():
            self.fitness = self._individual.fitness
        self.__cached_active_features_mask = None
        self.__len = hp_manager.active_features_mask_len(hyperparams=individual)

    def active_features_mask(self):
        if self.__cached_active_features_mask is None:
            self.__cached_active_features_mask = self.__hp_manager.active_features_mask(hyperparams=self._individual)
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
        return self._individual.get_predictors()

    def has_fitness(self):
        return self._individual.has_fitness()

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

    def _hp_manager(self) -> HyperparamManager:
        return self.__hp_manager

    def downlift(self, lifter: FeatureSpaceLifterMV) -> IndividualWithContext:
        predictive_individual = self._individual
        downlifted_predictive_individual =\
            PeculiarIndividualSparse(n_objectives=predictive_individual.n_objectives(),
                                     seq=lifter.collapse().downlift(self.active_features_mask()))
        downlifted_predictive_individual.fitness = copy(predictive_individual.fitness)
        downlifted_predictive_individual.set_std_dev(predictive_individual.std_dev())
        downlifted_predictive_individual.set_ci95(predictive_individual.ci95())
        downlifted_predictive_individual.set_bootstrap_mean(predictive_individual.bootstrap_mean())
        predictors = []
        for p in predictive_individual.get_predictors():
            if p is None:
                predictors.append(None)
            else:
                predictors.append(p.downlift(lifter=lifter))
        downlifted_predictive_individual.set_predictors(predictors=predictors)
        return IndividualWithContext(
            individual=downlifted_predictive_individual,
            hp_manager=DummyHpManager())  # The individual is now a Boolean sequence, so the dummy is fine.

    def modifiable_copy(self) -> PeculiarIndividual:
        res = PeculiarIndividualSparse(n_objectives=self.n_objectives(), seq=self)
        res.fitness = deepcopy(self.fitness)
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
