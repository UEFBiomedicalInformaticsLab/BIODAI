from abc import ABC, abstractmethod
from multiprocessing import Pool


class Mapper(ABC):

    @abstractmethod
    def map(self, func, iterable):
        raise NotImplementedError()


class MapperByMap(Mapper):
    __pool: Pool

    def __init__(self):
        raise NotImplementedError()

    def __init_workers_pool_if_needed(self, verbose=False):
        raise NotImplementedError()

    def map(self, func, iterable):
        raise NotImplementedError()
