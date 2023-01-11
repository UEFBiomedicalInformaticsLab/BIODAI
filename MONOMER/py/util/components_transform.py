from abc import abstractmethod, ABC
from collections import Sequence

from util.sparse_bool_list_by_set import top_k_mask


class ComponentsTransform(ABC):

    @abstractmethod
    def apply(self, x: Sequence[float]) -> Sequence[float]:
        """The len of the passed sequence does not change."""
        raise NotImplementedError()


class IdentityComponentsTransform(ComponentsTransform):

    def apply(self, x: Sequence[float]) -> Sequence[float]:
        return x


class HigherKComponents(ComponentsTransform):
    """Only the highest k components are kept (keeping also their initial value), all others are set to zero.
    Uses stable sorting to choose the highest k. In case of ties the first in the initial ordering has precedence."""
    __k: int

    def __init__(self, k: int):
        if k < 0:
            raise ValueError()
        self.__k = k

    def apply(self, x: Sequence[float]) -> Sequence[float]:
        top_mask = top_k_mask(elems=x, k=self.__k)
        res = [xi if bool(mi) else 0 for xi, mi in zip(x, top_mask)]
        return res
