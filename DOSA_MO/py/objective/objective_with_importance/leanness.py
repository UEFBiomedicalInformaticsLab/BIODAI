import math
from typing import Union

from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from objective.objective_with_importance.structural_objective_computer_with_importance import \
    StructuralObjectiveComputerWithImportance


class Leanness(StructuralObjectiveComputerWithImportance):

    @staticmethod
    def requires_target() -> bool:
        return False

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        if hp_manager is None:
            raise ValueError("hp_manager is None")
        n_features = hp_manager.n_active_features(hyperparams)
        return CVResult(fitness=self.compute_from_n_features(n_features))

    @staticmethod
    def compute_from_n_features(n_features: int) -> float:
        return 1/(1+n_features)

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        return self.compute_from_structure_with_importance(
            hyperparams=hyperparams, hp_manager=hp_manager, x=None, y=None)

    def nick(self):
        return "leanness"

    @staticmethod
    def val_to_label_float(value) -> float:
        """ Not rounded to int since passed value can be an average. """
        if value <= 0.0:
            return math.inf
        else:
            return (1.0 / value) - 1.0


class SoftLeanness(StructuralObjectiveComputerWithImportance):

    @staticmethod
    def requires_target() -> bool:
        return False

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        if hp_manager is None:
            raise ValueError("hp_manager is None")
        n_features = hp_manager.n_active_features(hyperparams)
        return CVResult(fitness=self.compute_from_n_features(n_features))

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        return self.compute_from_structure_with_importance(hyperparams=hyperparams, hp_manager=hp_manager, x=None, y=None)

    def nick(self):
        return "soft_leanness"

    def name(self):
        return "soft leanness"

    @staticmethod
    def val_to_label_float(value) -> float:
        """ Not rounded to int since passed value can be an average. """
        if value <= 0.0:
            return math.inf
        else:
            return ((1.0 / value) - 1.0)**2

    @staticmethod
    def compute_from_n_features(n_features: int) -> float:
        return 1.0/(1.0+math.sqrt(n_features))

    @staticmethod
    def leanness_to_soft_leanness(leanness: float) -> float:
        return SoftLeanness.compute_from_n_features(round(Leanness().val_to_label_float(leanness)))


class RootLeanness(StructuralObjectiveComputerWithImportance):

    @staticmethod
    def requires_target() -> bool:
        return False

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None], x: DataFrame,
                               y: DataFrame) -> CVResult:
        if hp_manager is None:
            raise ValueError("hp_manager is None")
        n_features = hp_manager.n_active_features(hyperparams)
        return CVResult(fitness=self.compute_from_n_features(n_features))

    def compute_from_classes(self, hyperparams, hp_manager: HyperparamManager,
                             test_pred, test_true,
                             train_pred=None, train_true=None) -> CVResult:
        return self.compute_from_structure_with_importance(hyperparams=hyperparams, hp_manager=hp_manager, x=None, y=None)

    def nick(self):
        return "root_leanness"

    def name(self):
        return "root leanness"

    @staticmethod
    def val_to_label_float(value) -> float:
        """ Not rounded to int since passed value can be an average. """
        if value <= 0.0:
            return math.inf
        else:
            return (1.0 / (value**2)) - 1.0

    @staticmethod
    def compute_from_n_features(n_features: int) -> float:
        return math.sqrt(Leanness.compute_from_n_features(n_features=n_features))
