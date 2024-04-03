from abc import ABC, abstractmethod
from typing import Optional

from ga_components.mut_flip_bit import mut_flip_bit
from ga_components.personalized_mutation import personalized_mutation
from ga_components.symmetric_flip import symmetric_flip_with_mask
from individual.Individual import Individual
from util.named import NickNamed
from util.summable import SummableSequence


class BitlistMutation(NickNamed, ABC):

    @abstractmethod
    def mutate(self, individual: Individual, frequency: float, active_mask=None) -> tuple[Individual]:
        """Works in place. Returns a sequence of 1 individual for DEAP compatibility."""
        raise NotImplementedError()

    @abstractmethod
    def uses_personalized_feature_importance(self) -> bool:
        """A feature importance computed on the specific individual."""
        raise NotImplementedError()


class FlipMutation(BitlistMutation):

    def mutate(self, individual: Individual, frequency: float, active_mask=None) -> tuple[Individual]:
        len_ind = len(individual)
        if len_ind > 0:
            indpb = frequency / len_ind  # Probability of flipping a bit.
            # Probability of flipping a bit does not take into account the active_mask for historical reasons.
            return mut_flip_bit(individual=individual, indpb=indpb, mask=active_mask)
        else:
            return individual,

    def nick(self) -> str:
        return "flip"

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()

    def uses_personalized_feature_importance(self) -> bool:
        return False


class SymmetricFlipMutation(BitlistMutation):

    def mutate(self, individual: Individual,
               frequency: float, active_mask: Optional[SummableSequence] = None) -> tuple[Individual]:
        return symmetric_flip_with_mask(individual=individual, frequency=frequency, active_mask=active_mask)

    def nick(self) -> str:
        return "symm"

    def name(self) -> str:
        return "symmetric flip"

    def __str__(self) -> str:
        return self.name()

    def uses_personalized_feature_importance(self) -> bool:
        return False


class PersonalizedMutation(BitlistMutation):

    def mutate(self, individual: Individual, frequency: float,
               active_mask: Optional[SummableSequence] = None) -> tuple[Individual]:
        return personalized_mutation(individual=individual, frequency=frequency, active_mask=active_mask)

    def nick(self) -> str:
        return "pers"

    def name(self) -> str:
        return "personalized mutation"

    def __str__(self) -> str:
        return self.name()

    def uses_personalized_feature_importance(self) -> bool:
        return True
