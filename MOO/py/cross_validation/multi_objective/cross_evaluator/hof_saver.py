from pathlib import Path

from pandas import DataFrame, concat

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from path_utils import create_optimizer_save_path
from util.printer.printer import Printer


SOLUTION_FEATURES_STR = "solution_features"
SOLUTION_FEATURES_PREFIX = SOLUTION_FEATURES_STR + "_fold_"
SOLUTION_FEATURES_EXTENSION = ".csv"
SOLUTION_FITNESSES_STR = "solution_fitnesses"
SOLUTION_FITNESSES_FOLD_PREFIX = SOLUTION_FITNESSES_STR + "_fold_"
SOLUTION_FITNESSES_EXTENSION = ".csv"

SOLUTION_FITNESSES_FILE_NAME = SOLUTION_FITNESSES_STR + SOLUTION_FITNESSES_EXTENSION
# Used in external validation when there is just one solution

TEST_STR = "test"
TEST_PREFIX = TEST_STR + "_"
HOFS_STR = "hofs"


def save_hof_features(path_saves: str, file_name: str, feature_names: [str], hof: MultiObjectiveOptimizerResult):
    Path(path_saves).mkdir(parents=True, exist_ok=True)
    df = hof.individuals_to_df()
    df.columns = feature_names
    df = df.astype(int)
    df = df.loc[:, (df != 0).any(axis=0)]  # Remove features that are never used.
    file = path_saves + file_name
    df.to_csv(file, index=False)


def compute_all_fitnesses(hof: MultiObjectiveOptimizerResult,
                          objectives: [PersonalObjective],
                          x, y) -> DataFrame:
    n_objectives = len(objectives)
    fitnesses = dict()
    for o_i in range(n_objectives):
        objective = objectives[o_i]
        if objective.requires_predictions():
            predictors = hof.predictors_for_objective(objective_num=o_i)
            y_test = None
            if objective.has_outcome_label():
                y_test = y[objective.outcome_label()]
            res_i = objective.compute_from_predictor_and_test_all(predictors=predictors, x_test=x, y_test=y_test)
        else:
            res_i = objective.compute_from_hyperparams_all(hyperparams_seq=hof.hyperparams())
        fitnesses[objective.nick()] = res_i
    return DataFrame(fitnesses)


def save_hof_fitnesses(path_saves: str, file_name: str,
                       hof: MultiObjectiveOptimizerResult,
                       objectives: [PersonalObjective],
                       x_train, y_train, x_test=None, y_test=None,
                       train_name: str = "train", cv_name: str = "inner_cv", test_name: str = TEST_STR):
    Path(path_saves).mkdir(parents=True, exist_ok=True)
    file = path_saves + file_name
    df = compute_all_fitnesses(hof=hof, objectives=objectives, x=x_train, y=y_train).add_prefix(str(train_name) + "_")
    if hof.has_fitnesses():
        inner_cv_df = hof.fitnesses_to_df()
        inner_cv_df.columns = [o.nick() for o in objectives]
        inner_cv_df = inner_cv_df.add_prefix(str(cv_name) + "_")
        df = concat((inner_cv_df, df), axis=1)
    if x_test is not None and y_test is not None:
        test_df = compute_all_fitnesses(
            hof=hof, objectives=objectives, x=x_test, y=y_test).add_prefix(TEST_PREFIX)
        df = concat((df, test_df), axis=1)
    df.to_csv(path_or_buf=file, index=False)


def hofs_path_from_optimizer_path(optimizer_path: str) -> str:
    return optimizer_path + HOFS_STR + "/"


def main_path_for_saves(base_path: str, optimizer_nick: str) -> str:
    optimizer_path = create_optimizer_save_path(save_path=base_path, optimizer_nick=optimizer_nick)
    return hofs_path_from_optimizer_path(optimizer_path=optimizer_path)


def path_for_saves(base_path: str, optimizer_nick: str, hof_nick: str) -> str:
    return main_path_for_saves(base_path=base_path, optimizer_nick=optimizer_nick) + hof_nick + "/"


def solution_features_file_name(fold_index: int) -> str:
    return SOLUTION_FEATURES_PREFIX + str(fold_index) + SOLUTION_FEATURES_EXTENSION


def solution_fitnesses_file_name(fold_index: int) -> str:
    return SOLUTION_FITNESSES_FOLD_PREFIX + str(fold_index) + SOLUTION_FITNESSES_EXTENSION


class HofsSaver(MultiObjectiveCrossEvaluator):
    __save_path: str
    __objectives: [PersonalObjective]

    def __init__(self, save_path: str, objectives: [PersonalObjective]):
        self.__save_path = save_path
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):
        n_folds = len(non_dominated_predictors_with_hyperparams)
        if n_folds == 0:
            return None
        hof_nick = non_dominated_predictors_with_hyperparams[0].nick()
        path_saves = path_for_saves(
            base_path=self.__save_path, optimizer_nick=optimizer_nick, hof_nick=hof_nick)
        printer.print_variable(var_name="Path for hall of fames", var_value=path_saves)
        feature_names = input_data.collapsed_feature_names()

        for fold in range(n_folds):
            hof = non_dominated_predictors_with_hyperparams[fold]
            save_hof_features(
                path_saves=path_saves,
                file_name=solution_features_file_name(fold_index=fold),
                feature_names=feature_names,
                hof=hof)
            x_train, y_train, x_test, y_test =\
                input_data.select_all_sets(
                    train_indices=folds.train_indices(fold), test_indices=folds.test_indices(fold))
            save_hof_fitnesses(path_saves=path_saves,
                               file_name=solution_fitnesses_file_name(fold_index=fold),
                               hof=hof,
                               objectives=self.__objectives,
                               x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def name(self) -> str:
        return "Hall of fames saver"
