from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence

from individual.fit_individual import FitIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from model.multi_view.mv_predictor import MVPredictor
from model.sv_model import SVPredictor, Predictor


class PredictiveIndividual(FitIndividual, ABC):

    def __init__(self, fitness: HighBestFitness):
        FitIndividual.__init__(self, fitness=fitness)

    @abstractmethod
    def get_predictors(self) -> Sequence[Predictor]:
        """Data should be prepared with this individual and also the appropriate hyperparameter manager
        before been fed to these predictors."""
        raise NotImplementedError()


class PredictiveIndividualSV(PredictiveIndividual, ABC):

    def __init__(self, fitness: HighBestFitness):
        PredictiveIndividual.__init__(self, fitness=fitness)

    @abstractmethod
    def get_predictors(self) -> Sequence[SVPredictor]:
        """Data should be prepared with this individual and also the appropriate hyperparameter manager
        before been fed to these predictors."""
        raise NotImplementedError()


class PredictiveIndividualMV(PredictiveIndividual, ABC):

    def __init__(self, fitness: HighBestFitness):
        PredictiveIndividual.__init__(self, fitness=fitness)

    @abstractmethod
    def get_predictors(self) -> Sequence[MVPredictor]:
        """Data should be prepared with this individual and also the appropriate hyperparameter manager
        before been fed to these predictors."""
        raise NotImplementedError()

