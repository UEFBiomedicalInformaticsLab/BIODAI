from typing import Iterable, Any

import numpy as np
from numpy import ndarray

from util.list_like import ListLike
from util.utils import IllegalStateError


class UniformList(ListLike):
    __value: Any
    __size: int

    def __init__(self, value: Any, size: int):
        self.__value = value
        if isinstance(size, (int, np.integer)) and size >= 0:
            self.__size = int(size)

    def __len__(self):
        return self.__size

    def __get_one_item(self, key: int) -> Any:
        if key < 0 or key >= self.__size:
            raise ValueError()
        return self.__value

    def __getitem__(self, key):
        """Supports integer key or slice key."""
        if isinstance(key, (int, np.integer)):
            return self.__get_one_item(key=key)
        if isinstance(key, slice):
            indices = range(*key.indices(self.__size))
            return [self.__get_one_item(key=i) for i in indices]  # Could return a uniform list for efficiency.
        raise TypeError()

    def __setitem__(self, key, value):
        raise IllegalStateError()

    def extend(self, iterable: Iterable):
        raise IllegalStateError()

    def to_numpy(self) -> ndarray:
        return np.full(shape=len(self), fill_value=self.__value)

    def sum(self):
        return self.__value * self.__size

    def append(self, value):
        raise IllegalStateError()
