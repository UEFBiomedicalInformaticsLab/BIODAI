from abc import ABC, abstractmethod
from random import random
from random import gauss

from util.list_like import ListLike


class ParticleMutation(ABC):

    @abstractmethod
    def mutate(self, particle: ListLike, completion: float):
        """Mutates in place. Completion is the completion of the optimizer between 0 and 1."""
        raise NotImplementedError()


class DummyParticleMutation(ParticleMutation):

    def mutate(self, particle: ListLike, completion: float):
        pass


class ConstantParticleMutation(ParticleMutation):
    __std_dev: float

    def __init__(self, std_dev: float):
        self.__std_dev = std_dev

    def mutate(self, particle: ListLike, completion: float):
        size = len(particle)
        if size > 0:
            m_prob = 1.0/size
            for i in range(size):
                if random() < m_prob:
                    particle[i] = gauss(mu=particle[i], sigma=self.__std_dev)


class SlowingParticleMutation(ParticleMutation):
    __std_dev: float

    def __init__(self, std_dev: float):
        self.__std_dev = std_dev

    def mutate(self, particle: ListLike, completion: float):
        size = len(particle)
        if size > 0:
            m_prob = 1.0 / size
            for i in range(size):
                if random() < m_prob:
                    particle[i] = gauss(mu=particle[i], sigma=self.__std_dev*(1.0-completion))