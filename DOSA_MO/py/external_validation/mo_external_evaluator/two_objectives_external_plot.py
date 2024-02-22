from collections.abc import Sequence
from pathlib import Path

from cross_validation.cross_validation import validate_single_fold_and_objective
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from external_validation.mo_external_evaluator.mo_external_evaluator import MultiObjectiveExternalEvaluator
from individual.fit_individual import get_fitnesses
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.plot_results import save_multiclass_scatter
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


class TwoObjectivesExternalPlot(MultiObjectiveExternalEvaluator):
    """Plots the objectives pairwise."""

    @staticmethod
    def __evaluate_pair(predictors_with_hyperparams: MultiObjectiveOptimizerResult,
                        input_data: InputData,
                        external_data: InputData,
                        path_for_plots: str,
                        objective_index_0: int,
                        objective_index_1: int,
                        objective_0: PersonalObjective,
                        objective_1: PersonalObjective,
                        printer: Printer):

        hof_nick = predictors_with_hyperparams.nick()
        name_0 = objective_0.name()
        name_1 = objective_1.name()
        nick_0 = objective_0.nick()
        nick_1 = objective_1.nick()
        outcome_label_0 = None
        if objective_0.has_outcome_label():
            outcome_label_0 = objective_0.outcome_label()
        outcome_label_1 = None
        if objective_1.has_outcome_label():
            outcome_label_1 = objective_1.outcome_label()

        printer.print("Creating plot for " + name_0 + " vs " + name_1)

        x_train = input_data.x().as_cached()
        y_train = input_data.outcomes_data_dict()
        x_test = external_data.x().as_cached()
        y_test = external_data.outcomes_data_dict()

        predictors = predictors_with_hyperparams.predictors()
        hyperparams = predictors_with_hyperparams.hyperparams()

        predictors_0 = [p[objective_index_0] for p in predictors]
        predictors_1 = [p[objective_index_1] for p in predictors]

        y_train0 = None
        y_test0 = None
        if outcome_label_0 is not None:
            y_train0 = y_train[outcome_label_0]
            y_test0 = y_test[outcome_label_0]
        y_train1 = None
        y_test1 = None
        if outcome_label_1 is not None:
            y_train1 = y_train[outcome_label_1]
            y_test1 = y_test[outcome_label_1]
        validate_single_fold_objective_0 = validate_single_fold_and_objective(
            x_train, y_train0, x_test, y_test0,
            predictors_0, hyperparams, objective_0, compute_confidence=False)
        validate_single_fold_objective_1 = validate_single_fold_and_objective(
            x_train, y_train1, x_test, y_test1,
            predictors_1, hyperparams, objective_1, compute_confidence=False)
        objective_0_train_fold_list = validate_single_fold_objective_0.train()
        objective_0_test_fold_list = validate_single_fold_objective_0.test()
        objective_1_train_fold_list = validate_single_fold_objective_1.train()
        objective_1_test_fold_list = validate_single_fold_objective_1.test()
        if hyperparams[0].has_fitness():
            objective_0_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_0)
            objective_1_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_1)
        else:
            objective_0_inner_cv_fold = []
            objective_1_inner_cv_fold = []

        save_multiclass_scatter(
            x=[objective_0.vals_to_labels(objective_0_train_fold_list),
               objective_0.vals_to_labels(objective_0_inner_cv_fold),
               objective_0.vals_to_labels(objective_0_test_fold_list)],
            y=[objective_1.vals_to_labels(objective_1_train_fold_list),
               objective_1.vals_to_labels(objective_1_inner_cv_fold),
               objective_1.vals_to_labels(objective_1_test_fold_list)],
            save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + ".png",
            x_label=name_0, y_label=name_1, class_labels=["internal", "internal cv", "external"])

    def evaluate(self, input_data: InputData, external_data: InputData, objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer,
                 hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        path_for_plots = optimizer_save_path + "objective_pairs/"

        printer.print_variable(var_name="Path for plots", var_value=path_for_plots)
        Path(path_for_plots).mkdir(parents=True, exist_ok=True)

        n_objectives = len(objectives)

        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    self.__evaluate_pair(
                        predictors_with_hyperparams=optimizer_result,
                        input_data=input_data,
                        external_data=external_data,
                        path_for_plots=path_for_plots,
                        objective_index_0=i, objective_index_1=j,
                        objective_0=objectives[i], objective_1=objectives[j],
                        printer=printer)

    def name(self) -> str:
        return "two-objectives external plotter"
