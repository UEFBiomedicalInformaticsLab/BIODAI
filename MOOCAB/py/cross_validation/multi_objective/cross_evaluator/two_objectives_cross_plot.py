from collections.abc import Sequence
from pathlib import Path
from typing import Union

from matplotlib import pyplot as plt

from cross_validation.cross_validation import validate_single_fold_and_objective
from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from individual.fit_individual import get_fitnesses
from individual.confident_individual import get_ci95s
from input_data.input_data import InputData
from objective.objective_with_importance.leanness import Leanness, RootLeanness, SoftLeanness
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from objective.social_objective import PersonalObjective
from util.hyperbox.hyperbox import Interval
from util.plot_results import save_multiclass_scatter, ADD_ELLIPSES_DEFAULT
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry


MIN_WIDTH = None


def substitute_missing_intervals(
        intervals: Sequence[Interval], fitnesses: Sequence[float]) -> list[Union[float, Interval]]:
    res = []
    for i, f in zip(intervals, fitnesses):
        if i is not None:
            res.append(i)
        else:
            res.append(f)
    return res


def substitute_missing_intervals_all(
        intervals: Sequence[Sequence[Interval]], fitnesses: Sequence[Sequence[float]]
) -> list[list[Union[float, Interval]]]:
    return [substitute_missing_intervals(intervals=i, fitnesses=f) for i, f in zip(intervals, fitnesses)]


def objective_priority(objective: PersonalObjective) -> int:
    nick = objective.objective_computer().nick()
    if (nick == Leanness().nick() or
            nick == SoftLeanness().nick() or
            nick == RootLeanness().nick()):
        return 1
    if objective.is_structural():
        return 2
    if objective.is_survival():
        return 3
    if objective.is_class_based():
        return 4
    return 5  # Just in case.


class TwoObjectivesCrossPlot(MultiObjectiveCrossEvaluator):
    """Plots the objectives pairwise."""
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __save_path: str

    def __init__(self, objectives: list[PersonalObjectiveWithImportance], save_path: str):
        self.__objectives = objectives
        self.__save_path = save_path

    def __evaluate_pair(self, predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult],
                        input_data: InputData, folds: Folds, path_for_plots: str,
                        objective_index_0: int, objective_index_1: int, printer: Printer,
                        interpolate: bool = True, add_ellipses: bool = ADD_ELLIPSES_DEFAULT):
        hof_nick = predictors_with_hyperparams[0].nick()
        objective_0 = self.__objectives[objective_index_0]
        objective_1 = self.__objectives[objective_index_1]
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

        printer.print("Creating plots for " + name_0 + " vs " + name_1)
        printer.print("Creating plots for each fold")

        use_inner_cv = True  # Will be set to false if not possible.
        objective_0_inner_cv_list = []
        objective_1_inner_cv_list = []
        objective_0_inner_cv_ci_list = []
        objective_1_inner_cv_ci_list = []
        objective_0_train_list = []
        objective_1_train_list = []
        objective_0_train_ci_list = []
        objective_1_train_ci_list = []
        objective_0_test_list = []
        objective_1_test_list = []
        objective_0_test_ci_list = []
        objective_1_test_ci_list = []
        for i in range(folds.n_folds()):

            x_train, y_train, x_test, y_test =\
                input_data.select_all_sets(
                    train_indices=folds.train_indices(fold_number=i),
                    test_indices=folds.test_indices(fold_number=i))
            x_train = x_train.as_cached()
            x_test = x_test.as_cached()
            non_dominated_predictors_with_hyperparams_i = predictors_with_hyperparams[i]
            predictors = non_dominated_predictors_with_hyperparams_i.predictors()
            hyperparams = non_dominated_predictors_with_hyperparams_i.hyperparams()

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
                predictors_0, hyperparams, objective_0, compute_confidence=True)
            validate_single_fold_objective_1 = validate_single_fold_and_objective(
                x_train, y_train1, x_test, y_test1,
                predictors_1, hyperparams, objective_1, compute_confidence=True)
            objective_0_train_fold_list = validate_single_fold_objective_0.train()
            objective_0_train_fold_ci_list = validate_single_fold_objective_0.train_ci()
            objective_0_test_fold_list = validate_single_fold_objective_0.test()
            objective_0_test_fold_ci_list = validate_single_fold_objective_0.test_ci()
            objective_1_train_fold_list = validate_single_fold_objective_1.train()
            objective_1_train_fold_ci_list = validate_single_fold_objective_1.train_ci()
            objective_1_test_fold_list = validate_single_fold_objective_1.test()
            objective_1_test_fold_ci_list = validate_single_fold_objective_1.test_ci()
            if use_inner_cv and hyperparams[0].has_fitness():
                objective_0_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_0)
                objective_0_inner_cv_fold_ci = get_ci95s(pop=hyperparams, fitness_index=objective_index_0)
                objective_1_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_1)
                objective_1_inner_cv_fold_ci = get_ci95s(pop=hyperparams, fitness_index=objective_index_1)
            else:
                objective_0_inner_cv_fold = []
                objective_1_inner_cv_fold = []
                objective_0_inner_cv_fold_ci = []
                objective_1_inner_cv_fold_ci = []
                use_inner_cv = False
            objective_0_inner_cv_list.extend(objective_0_inner_cv_fold)
            objective_1_inner_cv_list.extend(objective_1_inner_cv_fold)
            objective_0_inner_cv_ci_list.extend(objective_0_inner_cv_fold_ci)
            objective_1_inner_cv_ci_list.extend(objective_1_inner_cv_fold_ci)
            objective_0_train_list.extend(objective_0_train_fold_list)
            objective_0_train_ci_list.extend(objective_0_train_fold_ci_list)
            objective_1_train_list.extend(objective_1_train_fold_list)
            objective_1_train_ci_list.extend(objective_1_train_fold_ci_list)
            objective_0_test_list.extend(objective_0_test_fold_list)
            objective_0_test_ci_list.extend(objective_0_test_fold_ci_list)
            objective_1_test_list.extend(objective_1_test_fold_list)
            objective_1_test_ci_list.extend(objective_1_test_fold_ci_list)

            save_multiclass_scatter(
                x=[objective_0.vals_to_labels(objective_0_train_fold_list),
                   objective_0.vals_to_labels(objective_0_inner_cv_fold),
                   objective_0.vals_to_labels(objective_0_test_fold_list)],
                y=[objective_1.vals_to_labels(objective_1_train_fold_list),
                   objective_1.vals_to_labels(objective_1_inner_cv_fold),
                   objective_1.vals_to_labels(objective_1_test_fold_list)],
                save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + "_fold_" + str(i) + ".png",
                x_label=name_0, y_label=name_1, class_labels=["train", "inner cv", "test"],
                interpolate=interpolate, add_ellipses=add_ellipses)

        printer.print("Creating plot for union of folds")
        obj0_lists = [
            objective_0.vals_to_labels(objective_0_train_list),
            objective_0.vals_to_labels(objective_0_inner_cv_list),
            objective_0.vals_to_labels(objective_0_test_list)]
        obj1_lists = [
            objective_1.vals_to_labels(objective_1_train_list),
            objective_1.vals_to_labels(objective_1_inner_cv_list),
            objective_1.vals_to_labels(objective_1_test_list)]
        save_multiclass_scatter(
            x=obj0_lists,
            y=obj1_lists,
            save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + ".png",
            x_label=name_0, y_label=name_1, class_labels=["train", "inner cv", "test"],
            interpolate=interpolate, add_ellipses=add_ellipses)

        printer.print("Creating plot of confidence intervals")
        obj0_ci_lists = [
            objective_0.vals_to_labels(objective_0_train_ci_list),
            objective_0.vals_to_labels(objective_0_inner_cv_ci_list),
            objective_0.vals_to_labels(objective_0_test_ci_list)]
        obj1_ci_lists = [
            objective_1.vals_to_labels(objective_1_train_ci_list),
            objective_1.vals_to_labels(objective_1_inner_cv_ci_list),
            objective_1.vals_to_labels(objective_1_test_ci_list)]
        save_multiclass_scatter(
            x=substitute_missing_intervals_all(intervals=obj0_ci_lists, fitnesses=obj0_lists),
            y=substitute_missing_intervals_all(intervals=obj1_ci_lists, fitnesses=obj1_lists),
            save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + "_ci.png",
            x_label=name_0, y_label=name_1, class_labels=["train", "inner cv", "test"],
            min_width=MIN_WIDTH,
            interpolate=interpolate, add_ellipses=add_ellipses)

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry()):

        path_for_plots = self.__save_path + optimizer_nick + "/" + "objective_pairs/"

        printer.print_variable(var_name="Path for plots", var_value=path_for_plots)
        Path(path_for_plots).mkdir(parents=True, exist_ok=True)

        n_objectives = len(self.__objectives)

        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    priority_i = objective_priority(objective=self.__objectives[i])
                    priority_j = objective_priority(objective=self.__objectives[j])
                    if priority_i <= priority_j:
                        try:
                            interpolate = (priority_i <= 1)
                            self.__evaluate_pair(
                                predictors_with_hyperparams=non_dominated_predictors_with_hyperparams,
                                input_data=input_data, folds=folds, path_for_plots=path_for_plots,
                                objective_index_0=i, objective_index_1=j, printer=printer,
                                interpolate=interpolate)
                        except BaseException as e:
                            printer.print(
                                "Evaluation of pair of objectives failed with the following exception.\n" +
                                str(e) + "\n" +
                                "Objectives: " + self.__objectives[i].name() + " and " +
                                self.__objectives[j].name() + "\n" +
                                "The evaluator " + self.name() + " will try to continue.")
                            plt.close("all")  # In case some fig is not closed properly.

    def name(self) -> str:
        return "two-objectives cross-plotter"
