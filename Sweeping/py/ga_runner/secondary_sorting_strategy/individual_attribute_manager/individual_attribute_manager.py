from abc import ABC, abstractmethod
from collections.abc import Sequence

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual


class IndividualAttributeManager(ABC):
    """Manager for individual attributes like crowding distance or social space."""

    @abstractmethod
    def attribute_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, individuals: Sequence[PeculiarIndividual], hp_manager: HyperparamManager):
        raise NotImplementedError()

    @staticmethod
    def add_to_stats() -> bool:
        return False

    def getter(self):
        return self.__class__.get

    @staticmethod
    def get(ind: PeculiarIndividual) -> float:
        raise NotImplementedError()

    def __str__(self):
        return "Attribute manager for " + self.attribute_name()
