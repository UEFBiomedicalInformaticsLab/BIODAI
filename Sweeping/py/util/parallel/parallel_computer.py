from abc import ABC, abstractmethod
from typing import Any, Optional

from util.parallel.parallel_result import ParallelResult


class ParallelComputer(ABC):

    @abstractmethod
    def compute(self, common_data: Any, job: Any) -> ParallelResult:
        raise NotImplementedError()

    def error_message(self, job: Any) -> Optional[str]:
        """Override to provide an additional string to the error report when there is an exception."""
        return None
