from collections.abc import Sequence
from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR, \
    save_hof_confusions
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from external_validation.mo_external_evaluator.mo_external_evaluator import MultiObjectiveExternalEvaluator
from hall_of_fame.hof_utils import hof_path
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


class ExternalConfusionMatricesSaver(MultiObjectiveExternalEvaluator):
    __objectives: Sequence[PersonalObjective]

    def __init__(self, objectives: list[PersonalObjective]):
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, external_data: InputData, objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        hof_p = hof_path(optimizer_save_path=optimizer_save_path, hof_nick=optimizer_result.nick())
        path_saves = hof_p + CONFUSION_MATRIX_STR + "/"
        fold_test_data = external_data
        save_hof_confusions(path_saves=path_saves,
                            fold_index=None,
                            hof=optimizer_result,
                            objectives=self.__objectives,
                            fold_test_data=fold_test_data)

    def name(self) -> str:
        return "External confusion matrices saver"
