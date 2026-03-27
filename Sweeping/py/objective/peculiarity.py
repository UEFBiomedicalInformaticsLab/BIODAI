from typing import Optional, Any

from sklearn.metrics import mean_absolute_error

from cross_validation.single_objective.cv_result import CVResult

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from model.multi_view.multi_view_model import MVModel
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from objective.social_objective_factory import SocialObjectiveFactory
from objective.social_objective import SocialObjective
from util.uniform_list import UniformList
from util.utils import IllegalStateError


PECULIARITY_NAME = "peculiarity"


class Peculiarity(SocialObjective):

    def __init__(self):
        self.__average = None
        self.__average_sum = None

    def update(self, hp_pop):
        self.__average = [sum(col)/len(col) for col in zip(*hp_pop)]
        self.__average_sum = sum(self.__average)

    def compute_from_classes(
            self, test_pred, test_true,
            train_pred=None, train_true=None,
            hyperparams: Optional[Any] = None,
            hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        if self.__average is None:
            raise ValueError("Calling compute before update.")
        mae = mean_absolute_error(self.__average, hyperparams)
        denominator = max(self.__average_sum, sum(hyperparams))
        return CVResult(fitness=mae / denominator,
                        importances=UniformList(value=0.0, size=hp_manager.n_predictive_features(hyperparams=hyperparams)))

    def requires_predictions(self):
        return False

    def name(self):
        return PECULIARITY_NAME

    def __str__(self):
        return self.name()

    def objective_computer(self) -> ObjectiveComputerWithImportance:
        raise IllegalStateError()

    def mv_model(self) -> MVModel:
        raise IllegalStateError()


class PeculiarityFactory(SocialObjectiveFactory):

    def create(self) -> SocialObjective:
        return Peculiarity()

    def nick(self) -> str:
        return PECULIARITY_NAME
