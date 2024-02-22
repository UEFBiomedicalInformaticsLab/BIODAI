from abc import ABC, abstractmethod

from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput


class FitnessAdjuster(ABC):

    @abstractmethod
    def adjust_fitness(self, input_data: FitnessAdjusterInput) -> float:
        raise NotImplementedError()


class DummyFitnessAdjuster(FitnessAdjuster):

    def adjust_fitness(self, input_data: FitnessAdjusterInput) -> float:
        return input_data.original_fitness()

    def __str__(self) -> str:
        return "dummy fitness adjuster"
