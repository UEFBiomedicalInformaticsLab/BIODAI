from abc import abstractmethod
from typing import NamedTuple

from deprecation import deprecated

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV
from individual.predictive_individual import PredictiveIndividualSV
from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from model.sv_model import SVPredictor
from util.named import NickNamed


@deprecated("Old SV implementation, remove when safe.")
class SingleObjectiveOptimizerResultSV(NamedTuple):
    predictor: SVPredictor
    hyperparams: PredictiveIndividualSV
    hp_manager: SvHyperparamManager


class SingleObjectiveOptimizerResultMV(NamedTuple):
    predictor: MVPredictor
    hyperparams: ConfidentPredictiveIndividualMV
    hp_manager: MvHyperparamManager


class SOOptimizer(NickNamed):

    @abstractmethod
    def optimize(self, input_data: EvaluationReadyInputData) -> SingleObjectiveOptimizerResultMV:
        raise NotImplementedError()
