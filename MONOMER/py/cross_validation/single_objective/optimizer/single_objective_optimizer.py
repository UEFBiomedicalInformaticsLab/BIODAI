from abc import abstractmethod
from typing import NamedTuple

from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.predictive_individual import PredictiveIndividual
from model.model import Classifier
from util.named import NickNamed


class SingleObjectiveOptimizerResult(NamedTuple):
    predictor: Classifier
    hyperparams: PredictiveIndividual
    hp_manager: HyperparamManager


class SOOptimizer(NickNamed):

    @abstractmethod
    def optimize(self, views, y) -> SingleObjectiveOptimizerResult:
        raise NotImplementedError()
