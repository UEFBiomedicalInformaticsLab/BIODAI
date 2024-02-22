from collections.abc import Sequence
from typing import Iterable

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.individual_with_context import IndividualWithContext
from individual.peculiar_individual import PeculiarIndividual
from model.model import Classifier
from util.list_like import ListLike
from util.utils import IllegalStateError


class PeculiarIndividualWithContext(PeculiarIndividual, IndividualWithContext):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified."""
    __cached_active_features_mask: ListLike
    __len: int

    def __init__(self, individual: PeculiarIndividual, hp_manager: HyperparamManager):
        IndividualWithContext.__init__(self=self, individual=individual, hp_manager=hp_manager)
        self.__cached_active_features_mask = None
        self.__len = hp_manager.active_features_mask_len(hyperparams=individual)

    def active_features_mask(self):
        if self.__cached_active_features_mask is None:
            self.__cached_active_features_mask = self._hp_manager().active_features_mask(hyperparams=self._individual)
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

    def __str__(self):
        ret_string = ""
        ret_string += str(self.fitness) + " "
        ret_string += str(self.active_features_mask())
        return ret_string

    def get_predictors(self) -> Sequence[Classifier]:
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

    def get_stat(self, name):
        return self._individual.get_stat(name=name)

    def get_stats(self):
        return self._individual.get_stats()

    def get_crowding_distance(self):
        return self._individual.get_crowding_distance()

    def get_peculiarity(self):
        return self._individual.get_peculiarity()

    def get_social_space(self):
        return self._individual.get_social_space()

    def __setitem__(self, key, value):
        raise IllegalStateError()

    def extend(self, iterable: Iterable):
        raise IllegalStateError()

    def append(self, value):
        raise IllegalStateError()


def contextualize(hp: PeculiarIndividual, hp_manager: HyperparamManager) -> PeculiarIndividualWithContext:
    return PeculiarIndividualWithContext(individual=hp, hp_manager=hp_manager)


def contextualize_all(hps: Iterable[PeculiarIndividual], hp_manager: HyperparamManager
                      ) -> list[PeculiarIndividualWithContext]:
    return [contextualize(h, hp_manager) for h in hps]


def contextualize_mothballed(hp: PeculiarIndividual, hp_manager: HyperparamManager) -> PeculiarIndividualWithContext:
    return PeculiarIndividualWithContext(individual=hp.mothball(), hp_manager=hp_manager)


def contextualize_all_mothballed(hps: Iterable[PeculiarIndividual], hp_manager: HyperparamManager
                                 ) -> list[PeculiarIndividualWithContext]:
    return [contextualize_mothballed(h, hp_manager) for h in hps]
