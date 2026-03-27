from abc import ABC
from typing import Optional, Any

from cross_validation.single_objective.cv_result import CVResult
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from util.utils import IllegalStateError


class StructuralObjectiveComputerWithImportance(ObjectiveComputerWithImportance, ABC):

    def _compute_with_kfold_cv_class_with_importance_mv(self, model: MVModel, data: ModelReadyInputData,
                                                        folds_list: list[tuple[IndexArray, IndexArray]],
                                                        compute_fi: bool = False,
                                                        compute_confidence: bool = False) -> CVResult:
        raise IllegalStateError()

    def compute_from_predictor_and_test_mv(self, predictor: MVPredictor, test_data: ModelReadyInputData,
                                           train_data: Optional[ModelReadyInputData] = None) -> CVResult:
        raise IllegalStateError()

    def is_crisp_class_objective_computer(self) -> bool:
        return False

    def is_proba_class_objective_computer(self) -> bool:
        return False

    def is_survival_objective_computer(self) -> bool:
        return False

    def is_structural_objective_computer(self) -> bool:
        return True

    @staticmethod
    def requires_training_predictions() -> bool:
        return False

    def requires_predictions(self) -> bool:
        """Requires actual predictions from a predictive model.
        An objective might require labels but not predictions."""
        return False

    def can_compute_from_classes(self) -> bool:
        return False

    def compute_from_classes_mv(self, test_pred, test_true, train_pred=None, train_true=None,
                                hyperparams: Optional[Any] = None,
                                hp_manager: Optional[MvHyperparamManager] = None) -> CVResult:
        raise IllegalStateError()
