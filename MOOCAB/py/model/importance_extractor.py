from abc import ABC, abstractmethod
from collections.abc import Sequence
from util.utils import IllegalStateError


class SklearnImportanceExtractor(ABC):

    @abstractmethod
    def can_extract_importance(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract_importance(self, sklearn_predictor) -> Sequence[float]:
        raise NotImplementedError()


class OffImportanceExtractor(SklearnImportanceExtractor):
    """Dummy coef extractor that cannot extract."""

    def can_extract_importance(self) -> bool:
        return False

    def extract_importance(self, sklearn_predictor) -> Sequence[float]:
        raise IllegalStateError()


class OnImportanceExtractor(SklearnImportanceExtractor, ABC):

    def can_extract_importance(self) -> bool:
        return True
