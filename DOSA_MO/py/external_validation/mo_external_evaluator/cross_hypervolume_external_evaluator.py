from collections.abc import Sequence

from cross_validation.multi_objective.cross_evaluator.cross_hypervolume_cross_evaluator import fold_hypervolumes
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from external_validation.mo_external_evaluator.mo_external_evaluator import MultiObjectiveExternalEvaluator
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer
from validation_registry.allowed_property_names import INNER_CV_HV_NAME, TRAIN_HV_NAME, TEST_HV_NAME, CROSS_HV_NAME
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


class CrossHypervolumeExternalEvaluator(MultiObjectiveExternalEvaluator):
    __objectives: Sequence[PersonalObjective]

    def __init__(self, objectives: list[PersonalObjective]):
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, external_data: InputData, objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):

        x_train = input_data.x().as_cached()
        x_test = external_data.x().as_cached()
        predictors = optimizer_result.predictors()
        hyperparams = optimizer_result.hyperparams()

        hvols = fold_hypervolumes(
            x_train=x_train, y_train=input_data.outcomes_data_dict(),
            x_test=x_test, y_test=external_data.outcomes_data_dict(),
            hyperparams=hyperparams, predictors=predictors, objectives=self.__objectives)
        printer.print("Hypervolumes")
        printer.print(str(hvols))
        if hvols.inner_cv_hypervolume is not None:
            hof_registry.set_property(name=INNER_CV_HV_NAME, value=hvols.inner_cv_hypervolume)
        if hvols.train_hypervolume is not None:
            hof_registry.set_property(name=TRAIN_HV_NAME, value=hvols.train_hypervolume)
        if hvols.test_hypervolume is not None:
            hof_registry.set_property(name=TEST_HV_NAME, value=hvols.test_hypervolume)
        if hvols.cross_hypervolume is not None:
            hof_registry.set_property(name=CROSS_HV_NAME, value=hvols.cross_hypervolume)

        return hvols

    def name(self) -> str:
        return "cross hypervolume"
