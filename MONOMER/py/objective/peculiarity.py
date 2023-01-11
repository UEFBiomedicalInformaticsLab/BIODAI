from sklearn.metrics import mean_absolute_error

from hyperparam_manager.hyperparam_manager import HyperparamManager
from objective.Social_objective_factory import SocialObjectiveFactory
from objective.social_objective import SocialObjective


class Peculiarity(SocialObjective):

    def __init__(self):
        self.__average = None
        self.__average_sum = None

    def update(self, hp_pop):
        self.__average = [sum(col)/len(col) for col in zip(*hp_pop)]
        self.__average_sum = sum(self.__average)

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager, test_pred, test_true, train_pred, train_true):
        if self.__average is None:
            raise ValueError("Calling compute before update.")
        mae = mean_absolute_error(self.__average, hyperparams)
        denominator = max(self.__average_sum, sum(hyperparams))
        return mae / denominator

    def requires_predictions(self):
        return False

    def is_dynamic(self):
        return True

    def name(self):
        return "peculiarity"

    def __str__(self):
        return self.name()


class PeculiarityFactory(SocialObjectiveFactory):
    def create(self) -> SocialObjective:
        return Peculiarity()
