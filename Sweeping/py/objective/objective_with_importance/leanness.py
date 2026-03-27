import math
from typing import Optional

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.individual_with_context import IndividualWithContext
from input_data.model_ready_input_data import ModelReadyInputData
from objective.objective_with_importance.structural_objective_computer_with_importance import \
    StructuralObjectiveComputerWithImportance

LEANNESS_NICK = "leanness"


def hyperparams_n_features(
        hyperparams,
        hp_manager: Optional[MvHyperparamManager] = None,
        data: Optional[ModelReadyInputData] = None) -> int:
    """Dirty workaround until we refactor Individual freeing it from backward compatibility with DEAP,
    avoiding exposing it as a sequence of Booleans."""
    if isinstance(hyperparams, IndividualWithContext):
        return hyperparams.sum()
    else:
        if hp_manager is None:
            if data is None:
                raise ValueError("Both hp_manager and data are None")
            else:
                hp_manager = MaskMvHpManager.create_from_input_data(input_data=data)
        return hp_manager.n_used_features(hyperparams=hyperparams)


class Leanness(StructuralObjectiveComputerWithImportance):

    @staticmethod
    def requires_target() -> bool:
        return False

    def compute_from_structure(self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
                               data: Optional[ModelReadyInputData] = None) -> CVResult:
        n_features = hyperparams_n_features(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            data=data)
        return CVResult(fitness=self.compute_from_n_features(n_features))

    @staticmethod
    def compute_from_n_features(n_features: int) -> float:
        return 1/(1+n_features)

    def nick(self):
        return LEANNESS_NICK

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

    def compute_from_structure(self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
                               data: Optional[ModelReadyInputData] = None) -> CVResult:
        n_features = hyperparams_n_features(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            data=data)
        return CVResult(fitness=self.compute_from_n_features(n_features))

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

    def compute_from_structure(self, hyperparams, hp_manager: Optional[MvHyperparamManager] = None,
                               data: Optional[ModelReadyInputData] = None) -> CVResult:
        n_features = hyperparams_n_features(
            hyperparams=hyperparams,
            hp_manager=hp_manager,
            data=data)
        return CVResult(fitness=self.compute_from_n_features(n_features))

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
