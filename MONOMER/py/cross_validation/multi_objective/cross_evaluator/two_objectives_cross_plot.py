from pathlib import Path

from cross_validation.cross_validation import validate_single_fold_and_objective
from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from individual.fit_individual import get_fitnesses
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.plot_results import save_multiclass_scatter
from util.printer.printer import Printer


class TwoObjectivesCrossPlot(MultiObjectiveCrossEvaluator):
    """Plots the objectives pairwise."""
    __objectives: [PersonalObjective]
    __save_path: str

    def __init__(self, objectives: list[PersonalObjective], save_path: str):
        self.__objectives = objectives
        self.__save_path = save_path

    def __evaluate_pair(self, predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                        input_data: InputData, folds: Folds, path_for_plots: str,
                        objective_index_0: int, objective_index_1: int, printer: Printer):
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
        objective_0_train_list = []
        objective_1_train_list = []
        objective_0_test_list = []
        objective_1_test_list = []
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
                predictors_0, hyperparams, objective_0)
            validate_single_fold_objective_1 = validate_single_fold_and_objective(
                x_train, y_train1, x_test, y_test1,
                predictors_1, hyperparams, objective_1)
            objective_0_train_fold_list = validate_single_fold_objective_0.objective_on_train
            objective_0_test_fold_list = validate_single_fold_objective_0.objective_on_test
            objective_1_train_fold_list = validate_single_fold_objective_1.objective_on_train
            objective_1_test_fold_list = validate_single_fold_objective_1.objective_on_test
            if use_inner_cv and hyperparams[0].has_fitness():
                objective_0_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_0)
                objective_1_inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index_1)
            else:
                objective_0_inner_cv_fold = []
                objective_1_inner_cv_fold = []
                use_inner_cv = False
            objective_0_inner_cv_list.extend(objective_0_inner_cv_fold)
            objective_1_inner_cv_list.extend(objective_1_inner_cv_fold)
            objective_0_train_list.extend(objective_0_train_fold_list)
            objective_1_train_list.extend(objective_1_train_fold_list)
            objective_0_test_list.extend(objective_0_test_fold_list)
            objective_1_test_list.extend(objective_1_test_fold_list)

            save_multiclass_scatter(
                x=[objective_0.vals_to_labels(objective_0_train_fold_list),
                   objective_0.vals_to_labels(objective_0_inner_cv_fold),
                   objective_0.vals_to_labels(objective_0_test_fold_list)],
                y=[objective_1.vals_to_labels(objective_1_train_fold_list),
                   objective_1.vals_to_labels(objective_1_inner_cv_fold),
                   objective_1.vals_to_labels(objective_1_test_fold_list)],
                save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + "_fold_" + str(i) + ".png",
                x_label=name_0, y_label=name_1, class_labels=["train", "inner cv", "test"])

        printer.print("Creating plot for union of folds")
        save_multiclass_scatter(
            x=[objective_0.vals_to_labels(objective_0_train_list),
               objective_0.vals_to_labels(objective_0_inner_cv_list),
               objective_0.vals_to_labels(objective_0_test_list)],
            y=[objective_1.vals_to_labels(objective_1_train_list),
               objective_1.vals_to_labels(objective_1_inner_cv_list),
               objective_1.vals_to_labels(objective_1_test_list)],
            save_file=path_for_plots + hof_nick + "_" + nick_0 + "_" + nick_1 + ".png",
            x_label=name_0, y_label=name_1, class_labels=["train", "inner cv", "test"])

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):

        path_for_plots = self.__save_path + optimizer_nick + "/" + "objective_pairs/"

        printer.print_variable(var_name="Path for plots", var_value=path_for_plots)
        Path(path_for_plots).mkdir(parents=True, exist_ok=True)

        n_objectives = len(self.__objectives)

        for i in range(n_objectives):
            for j in range(n_objectives):
                if i != j:
                    self.__evaluate_pair(
                        predictors_with_hyperparams=non_dominated_predictors_with_hyperparams,
                        input_data=input_data, folds=folds, path_for_plots=path_for_plots,
                        objective_index_0=i, objective_index_1=j, printer=printer)

    def name(self) -> str:
        return "two-objectives cross-plotter"
