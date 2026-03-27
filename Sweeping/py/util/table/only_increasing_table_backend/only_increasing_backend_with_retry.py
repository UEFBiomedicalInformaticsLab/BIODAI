import random
import time
from typing import Optional, Sequence
from numpy import ndarray
from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingTableBackend


def handle_exception(errors: list[str], e: Exception, delay: float, verbose: bool = True):
    """Does not alter the system random state."""
    attempt = len(errors) + 1
    error_msg = f"Attempt {attempt}: {type(e).__name__} - {str(e)}"
    if verbose:
        print(error_msg)
    errors.append(error_msg)
    rand = random.Random()
    delay = rand.expovariate(1 / delay)
    if verbose:
        print(f"Waiting for {delay:.2f} seconds before retrying.")
    time.sleep(delay)


def failure_error(errors: list[str]) -> RuntimeError:
    # Combine all error messages into one string
    error_summary = "\n".join(errors)
    return RuntimeError(f"Failed to read data after {len(errors)} retries. Errors:\n{error_summary}")


class OnlyIncreasingTableBackendWithRetry(OnlyIncreasingTableBackend):
    __inner: OnlyIncreasingTableBackend
    __retries: int
    __delay: float

    def __init__(self, inner: OnlyIncreasingTableBackend, retries: int = 5, delay: float = 2.0):
        self.__inner = inner
        self.__retries = retries
        self.__delay = delay

    def n_row(self) -> int:
        errors = []  # Store all exceptions
        for attempt in range(self.__retries):
            try:
                return self.__inner.n_row()
            except Exception as e:
                handle_exception(errors=errors, e=e, delay=self.__delay)
        raise failure_error(errors=errors)

    def n_col(self) -> int:
        errors = []  # Store all exceptions
        for attempt in range(self.__retries):
            try:
                return self.__inner.n_col()
            except Exception as e:
                handle_exception(errors=errors, e=e, delay=self.__delay)
        raise failure_error(errors=errors)

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        errors = []  # Store all exceptions
        for attempt in range(self.__retries):
            try:
                return self.__inner.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
            except Exception as e:
                handle_exception(errors=errors, e=e, delay=self.__delay)
        raise failure_error(errors=errors)

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        errors = []  # Store all exceptions
        for attempt in range(self.__retries):
            try:
                return self.__inner.colnames(selected=selected)
            except Exception as e:
                handle_exception(errors=errors, e=e, delay=self.__delay)
        raise failure_error(errors=errors)

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        errors = []  # Store all exceptions
        for attempt in range(self.__retries):
            try:
                return self.__inner.rownames(selected=selected)
            except Exception as e:
                handle_exception(errors=errors, e=e, delay=self.__delay)
        raise failure_error(errors=errors)

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()