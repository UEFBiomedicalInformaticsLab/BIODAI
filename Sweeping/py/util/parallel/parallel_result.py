from abc import ABC, abstractmethod
from typing import Any, Optional

from util.utils import IllegalStateError


class ParallelResult(ABC):

    @abstractmethod
    def result(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def error_message(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def is_correct(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_traceback(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def traceback(self) -> str:
        raise NotImplementedError()


class CorrectParallelResult(ParallelResult):
    __result: Any

    def __init__(self, result: Any):
        self.__result = result

    def result(self) -> Any:
        return self.__result

    def error_message(self) -> str:
        raise IllegalStateError()

    def is_correct(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self.result())

    def has_traceback(self) -> bool:
        return False

    def traceback(self) -> str:
        raise IllegalStateError()


class WrongParallelResult(ParallelResult):
    __message: str
    __trace: Optional[str]

    def __init__(self, message: str, traceback: Optional[str] = None):
        self.__message = message
        self.__trace = traceback

    def result(self) -> Any:
        raise IllegalStateError()

    def error_message(self) -> str:
        return self.__message

    def is_correct(self) -> bool:
        return False

    def has_traceback(self) -> bool:
        return self.__trace is not None

    def traceback(self) -> str:
        if self.has_traceback():
            return self.__trace
        else:
            raise IllegalStateError()

    def __str__(self) -> str:
        return str(self.error_message())