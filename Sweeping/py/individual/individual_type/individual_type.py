from abc import ABC, abstractmethod
from collections.abc import Sequence

from frozenlist import FrozenList

from individual.individual_type.gene_type import GeneType


class IndividualType(ABC):
    """An IndividualType must be immutable after creation."""
    __gene_types: FrozenList[GeneType]

    def __init__(self, gene_types: Sequence[GeneType]):
        self.__gene_types = FrozenList(items=gene_types)
        self.__gene_types.freeze()

    @abstractmethod
    def gene_types(self) -> Sequence[GeneType]:
        return self.__gene_types

    def n_genes(self) -> int:
        return len(self.__gene_types)

    def __eq__(self, other) -> bool:
        if isinstance(other, IndividualType):
            if hash(self) == hash(other):
                return self.gene_types() == other.gene_types()
            else:
                return False
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.__gene_types)  # FrozenList supports caching of the hash value.

    def __str__(self) -> str:
        return str(self.__gene_types)
