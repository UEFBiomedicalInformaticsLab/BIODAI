from abc import ABC, abstractmethod
from typing import Optional

from ga_components.mut_flip_bit import mut_flip_bit
from ga_components.symmetric_flip import symmetric_flip_with_mask
from util.named import NickNamed
from util.summable import SummableSequence


class BitlistMutation(NickNamed, ABC):

    @abstractmethod
    def mutate(self, individual: [], frequency: float, active_mask=None):
        raise NotImplementedError()


class FlipMutation(BitlistMutation):

    def mutate(self, individual: [], frequency: float, active_mask=None):
        indpb = frequency / len(individual)  # Probability of flipping a bit.
        # Probability of flipping a bit does not take into account the active_mask for historical reasons.
        return mut_flip_bit(individual=individual, indpb=indpb, mask=active_mask)

    def nick(self) -> str:
        return "flip"

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()


class SymmetricFlipMutation(BitlistMutation):

    def mutate(self, individual: [], frequency: float, active_mask: Optional[SummableSequence] = None):
        return symmetric_flip_with_mask(individual=individual, frequency=frequency, active_mask=active_mask)

    def nick(self) -> str:
        return "symm"

    def name(self) -> str:
        return "symmetric flip"

    def __str__(self) -> str:
        return self.name()
