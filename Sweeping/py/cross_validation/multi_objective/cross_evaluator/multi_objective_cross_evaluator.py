from collections.abc import Sequence
from typing import Optional

from cross_validation.folds import Folds
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.named import Named
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


class MultiObjectiveCrossEvaluator(Named):

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry(),
                 n_proc: Optional[int] = None):
        """
        Can return an object representing the result of the evaluation.
        non_dominated_predictors_with_hyperparams: A MultiObjectiveOptimizerResult for each fold.
        If n_cores is None, the evaluator will decide how many cores to use
        (typically all of them if it is parallelised).
        """
        raise NotImplementedError()


class DummyMOCrossEvaluator(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry(),
                 n_proc: Optional[int] = None):
        return None


class PrinterMOCrossEvaluator(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry(),
                 n_proc: Optional[int] = None):
        print("Non dominated predictors and hyperparameters for evaluation:")
        print(non_dominated_predictors_with_hyperparams)
        return None
