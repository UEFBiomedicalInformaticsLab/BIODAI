from abc import ABC, abstractmethod

from individual.fit_individual import FitIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from model.model import Classifier


class PredictiveIndividual(FitIndividual, ABC):

    def __init__(self, fitness: HighBestFitness):
        FitIndividual.__init__(self, fitness=fitness)

    @abstractmethod
    def get_predictors(self) -> [Classifier]:
        raise NotImplementedError()
