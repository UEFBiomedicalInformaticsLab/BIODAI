from cross_validation.folds import Folds, save_folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry

FOLDS_FILE_NAME = "folds.json"


class FoldsSaver(MultiObjectiveCrossEvaluator):
    __optimizer_save_path: str

    def __init__(self, optimizer_save_path: str):
        self.__optimizer_save_path = optimizer_save_path

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        save_folds(folds=folds, file_path=self.__optimizer_save_path + FOLDS_FILE_NAME)

    def name(self) -> str:
        return "fold saver"
