from abc import ABC, abstractmethod
from typing import Sequence

from util.utils import IllegalStateError


class SklearnCoefExtractor(ABC):

    @abstractmethod
    def can_extract_coef(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise NotImplementedError()


class OffCoefExtractor(SklearnCoefExtractor):
    """Dummy coef extractor that cannot extract."""

    def can_extract_coef(self) -> bool:
        return False

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        raise IllegalStateError()


class OnCoefExtractor(SklearnCoefExtractor, ABC):

    def can_extract_coef(self) -> bool:
        return True


class EmptyCoefExtractor(OnCoefExtractor):

    def extract_coef(self, sklearn_predictor) -> Sequence[Sequence[float]]:
        return []
