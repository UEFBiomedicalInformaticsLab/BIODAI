from abc import ABC, abstractmethod
from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.named import Named
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


class MultiObjectiveExternalEvaluator(Named, ABC):

    @abstractmethod
    def evaluate(self,
                 input_data: InputData,
                 external_data: InputData,
                 objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult,
                 optimizer_save_path: str,
                 printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        raise NotImplementedError()


class DummyMOExternalEvaluator(MultiObjectiveExternalEvaluator):

    def evaluate(self, input_data: InputData, external_data: InputData, objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        pass

    def name(self) -> str:
        return "dummy"
