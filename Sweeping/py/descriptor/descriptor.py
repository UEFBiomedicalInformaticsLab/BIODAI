from __future__ import annotations
from abc import ABC
from typing import Optional

from util.named import NickNamed


class Descriptor(NickNamed, ABC):
    """A descriptor describes a strategy to solve a problem. It can be represented by more than one string
    representations (to support different versions) and by slightly different callable objects (to support
    for different versions but also slightly different implementations leveraging different libraries, optimizations,
    etc. that are different in details that are not specified by the descriptor).
    On the opposite given a nickname or a callable object there is unambiguously only one related descriptor.
    The nickname and other strings returned directly by the descriptor refer to a default textual representation."""

    def __eq__(self, other) -> bool:
        if isinstance(other, Descriptor):
            return self.nick() == other.nick()
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.nick())


class Described(NickNamed):
    """We are not adding a static creator method that creates an instance starting from a descriptor because
    if the described algorithm is composite the composing parts might depend on an outside policy unknown
    by the current class."""
    __descriptor: Optional[Descriptor]

    def __init__(self, descriptor: Optional[Descriptor] = None):
        self.__descriptor = descriptor

    def _create_descriptor(self) -> Descriptor:
        """Override to provide on the fly creation of the descriptor.
        Not needed if it is received during initialization."""
        raise NotImplementedError("self type: " + str(type(self)))

    def descriptor(self) -> Descriptor:
        if self.__descriptor is None:
            self.__descriptor = self._create_descriptor()
        return self.__descriptor

    def __str__(self) -> str:
        """Override for more specific text."""
        return str(self.descriptor())

    def name(self) -> str:
        """Override for more specific text."""
        return self.descriptor().name()

    def nick(self) -> str:
        """Override for more specific text."""
        return self.descriptor().nick()
