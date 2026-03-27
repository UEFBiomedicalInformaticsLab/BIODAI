from abc import ABC
from typing import Iterable, Optional

from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from individual.confident_predictive_individual import ConfidentPredictiveIndividual, ConfidentPredictiveIndividualSV
from individual.individual_with_context import IndividualWithContextSV, IndividualWithContext
from individual.peculiar_individual import PeculiarIndividual


class PeculiarIndividualWithContext(PeculiarIndividual, IndividualWithContext, ABC):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified."""

    def __init__(self, individual: ConfidentPredictiveIndividual):
        IndividualWithContext.__init__(self=self, individual=individual)

    def __inner_peculiar_individual(self) -> PeculiarIndividual:
        """This is a workaround until we change the type of the inner individual."""
        individual = self._individual
        assert isinstance(individual, PeculiarIndividual)
        return individual

    def get_stat(self, name):
        return self.__inner_peculiar_individual().get_stat(name=name)

    def get_stats(self):
        return self.__inner_peculiar_individual().get_stats()

    def get_crowding_distance(self):
        return self.__inner_peculiar_individual().get_crowding_distance()

    def get_peculiarity(self) -> Optional[float]:
        return self.__inner_peculiar_individual().get_peculiarity()

    def get_social_space(self) -> Optional[float]:
        return self.__inner_peculiar_individual().get_social_space()


class PeculiarIndividualWithContextSV(PeculiarIndividualWithContext, IndividualWithContextSV):
    """Should be treated as unmodifiable, otherwise behaviour is unspecified."""

    def __init__(self, individual: ConfidentPredictiveIndividualSV, hp_manager: SvHyperparamManager):
        PeculiarIndividualWithContext.__init__(self=self, individual=individual)
        IndividualWithContextSV.__init__(self=self, individual=individual, hp_manager=hp_manager)


def contextualize(
        hp: ConfidentPredictiveIndividualSV, hp_manager: SvHyperparamManager) -> PeculiarIndividualWithContextSV:
    return PeculiarIndividualWithContextSV(individual=hp, hp_manager=hp_manager)


def contextualize_all(hps: Iterable[ConfidentPredictiveIndividualSV], hp_manager: SvHyperparamManager
                      ) -> list[PeculiarIndividualWithContextSV]:
    return [contextualize(h, hp_manager) for h in hps]
