from abc import ABC, abstractmethod

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike


class IndividualAttributeManager(ABC):
    """Manager for individual attributes like crowding distance or social space."""

    @abstractmethod
    def attribute_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, individuals: list[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
        raise NotImplementedError()

    @staticmethod
    def add_to_stats() -> bool:
        return False

    def getter(self):
        return self.__class__.get

    @staticmethod
    def get(ind: PeculiarIndividualByListlike) -> float:
        raise NotImplementedError()

    def __str__(self):
        return "Attribute manager for " + self.attribute_name()
