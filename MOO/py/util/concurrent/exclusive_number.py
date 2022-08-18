import os
import pathlib

from util.utils import try_make_file, IllegalStateError

EXCLUSIVE_NUMBERS_PATH = "./Exclusive_numbers/"
EXTENSION = ".tmp"


class ExclusiveNumber:
    """An object of this class can be used to acquire a unique number across multiple processes.
    It is intended to have a lifecycle inside a single process, it is not guaranteed to work correctly if
    copied to another process due multiple file deletions during exit."""
    __num: int

    def __init__(self):
        self.__num = -1

    @staticmethod
    def __path_str(num: int):
        return EXCLUSIVE_NUMBERS_PATH+str(num)+EXTENSION

    def __enter__(self) -> int:
        if self.__num != -1:
            raise IllegalStateError("Already holding an exclusive number.")
        n = 0
        created = False
        while not created:
            pathlib.Path(os.path.dirname(EXCLUSIVE_NUMBERS_PATH)).mkdir(parents=True, exist_ok=True)
            # Directory is created inside loop so that if another process deletes it we try again.
            created = try_make_file(filename=self.__path_str(n))
            if not created:
                n += 1
        self.__num = n
        return n

    def __exit_unchecked(self):
        path = pathlib.Path(self.__path_str(self.__num))
        self.__num = -1
        path.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__num == -1:
            raise IllegalStateError("Not holding an exclusive number.")
        self.__exit_unchecked()
        return False  # With false exceptions are not obscured.

    def __del__(self):
        if self.__num != -1:
            self.__exit_unchecked()
