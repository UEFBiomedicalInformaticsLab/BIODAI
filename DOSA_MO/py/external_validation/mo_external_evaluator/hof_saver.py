from typing import Sequence
from cross_validation.multi_objective.cross_evaluator.hof_saver import save_hof_features, \
    SOLUTION_FEATURES_STR, CSV_EXTENSION, save_hof_fitnesses_with_confidence
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from external_validation.mo_external_evaluator.mo_external_evaluator import MultiObjectiveExternalEvaluator
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry

SOLUTION_FEATURES_EXTERNAL = SOLUTION_FEATURES_STR + CSV_EXTENSION
TRAIN_STR = "internal"
EXTERNAL_STR = "external"
INTERNAL_CV_STR = "internal_cv"


class ExternalHofsSaver(MultiObjectiveExternalEvaluator):

    def evaluate(self, input_data: InputData, external_data: InputData,
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):

        hof_nick = optimizer_result.nick()
        path_saves = optimizer_save_path + "hofs/" + hof_nick + "/"
        printer.print_variable(var_name="Path for hall of fames", var_value=path_saves)
        feature_names = input_data.collapsed_feature_names()

        save_hof_features(
            path_saves=path_saves,
            file_name=SOLUTION_FEATURES_EXTERNAL,
            feature_names=feature_names,
            hof=optimizer_result)

        x_train = input_data.x().as_cached()
        y_train = input_data.outcomes_data_dict()
        x_test = external_data.x().as_cached()
        y_test = external_data.outcomes_data_dict()

        save_hof_fitnesses_with_confidence(
            path_saves=path_saves,
            fitnesses_file_name="solution_fitnesses.csv",
            std_dev_file_name="solution_std_devs.csv",
            ci_min_file_name="solution_ci_min.csv",
            ci_max_file_name="solution_ci_max.csv",
            hof=optimizer_result,
            objectives=objectives,
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            train_name=TRAIN_STR, cv_name=INTERNAL_CV_STR, test_name=EXTERNAL_STR)

    def name(self) -> str:
        return "Hall of fames saver"
