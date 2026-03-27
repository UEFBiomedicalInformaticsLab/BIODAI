from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from pandas import DataFrame, concat

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from location_manager.path_utils import path_for_saves
from util.printer.printer import Printer
from util.randoms import random_seed
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry

SOLUTION_STR = "solution"
SEP = "_"
FOLD_STR = "fold"
CSV_EXTENSION = ".csv"
SOLUTION_FEATURES_STR = SOLUTION_STR + SEP + "features"
SOLUTION_FEATURES_PREFIX = SOLUTION_FEATURES_STR + SEP + FOLD_STR + SEP
SOLUTION_FITNESSES_STR = SOLUTION_STR + SEP + "fitnesses"
SOLUTION_FITNESSES_FOLD_PREFIX = SOLUTION_FITNESSES_STR + SEP + FOLD_STR + SEP
SOLUTION_STD_DEVS_STR = SOLUTION_STR + SEP + "std_devs"
SOLUTION_STD_DEVS_FOLD_PREFIX = SOLUTION_STD_DEVS_STR + SEP + FOLD_STR + SEP
SOLUTION_CI_MIN_STR = SOLUTION_STR + SEP + "ci_min"
SOLUTION_CI_MIN_FOLD_PREFIX = SOLUTION_CI_MIN_STR + SEP + FOLD_STR + SEP
SOLUTION_CI_MAX_STR = SOLUTION_STR + SEP + "ci_max"
SOLUTION_CI_MAX_FOLD_PREFIX = SOLUTION_CI_MAX_STR + SEP + FOLD_STR + SEP


EXTERNAL_SOLUTION_FITNESSES_FILE_NAME = SOLUTION_FITNESSES_STR + CSV_EXTENSION
# Used in external validation when there is just one solution

TEST_STR = "test"
INNER_CV_STR = "inner_cv"
TEST_PREFIX = TEST_STR + "_"
INNER_CV_PREFIX = INNER_CV_STR + "_"


def save_hof_features(
        path_saves: str, file_name: str, feature_names: Sequence[str], hof: MultiObjectiveOptimizerResult):
    Path(path_saves).mkdir(parents=True, exist_ok=True)
    df = hof.individuals_to_df()
    if df.shape[1] != len(feature_names):
        raise ValueError(
            "Number of columns in dataframe from hof individuals is different to number of passed feature names.")
    df.columns = feature_names
    df = df.astype(int)
    df = df.loc[:, (df != 0).any(axis=0)]  # Remove features that are never used.
    file = path_saves + file_name
    df.to_csv(file, index=False)


def compute_all_fitnesses_with_confidence(
        hof: MultiObjectiveOptimizerResult,
        objectives: Sequence[PersonalObjectiveWithImportance],
        test_data: InputData,
        train_data: Optional[InputData] = None) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """First dataframe is for fitnesses, second for standard deviations. Then CI min and CI max."""
    n_objectives = len(objectives)
    fitnesses = dict()
    std_devs = dict()
    ci_mins = dict()
    ci_maxs = dict()
    for o_i in range(n_objectives):
        objective = objectives[o_i]
        objective_train_data = None
        if objective.requires_predictions():
            predictors = hof.predictors_for_objective(objective_num=o_i)
            outcome_label = objective.outcome_label()
            if train_data is not None:
                objective_train_data = train_data.model_ready(outcome=outcome_label)
            res_i_cv_results = objective.compute_from_predictor_and_test_with_importance_all_mv(
                predictors=predictors,
                test_data=test_data.model_ready(outcome=outcome_label),
                train_data=objective_train_data,
                compute_confidence=True, seed=random_seed())
        else:
            objective_computer = objective.objective_computer()
            hp_manager = MaskMvHpManager.create_from_input_data(input_data=test_data)
            if objective_computer.requires_target():
                outcome_label = objective.outcome_label()
                test_data = test_data.model_ready(outcome=outcome_label)
                res_i_cv_results = objective_computer.compute_from_structure_with_importance_all(
                    hyperparams_seq=hof.hyperparams(),
                    hp_manager=hp_manager,
                    data=test_data,
                    compute_fi=False, compute_confidence=True)
            else:
                res_i_cv_results = objective.compute_from_hyperparams_all(
                    hyperparams_seq=hof.hyperparams(), hp_manager=hp_manager)
        fitnesses[objective.nick()] = [r.fitness() for r in res_i_cv_results]
        std_devs[objective.nick()] = [r.std_dev() if r.has_std_dev() else None for r in res_i_cv_results]
        ci_mins[objective.nick()] = [r.ci95().a() if r.has_ci95() else None for r in res_i_cv_results]
        ci_maxs[objective.nick()] = [r.ci95().a() if r.has_ci95() else None for r in res_i_cv_results]
    return DataFrame(fitnesses), DataFrame(std_devs), DataFrame(ci_mins), DataFrame(ci_maxs)


def save_hof_fitnesses_with_confidence(
        path_saves: str,
        fitnesses_file_name: str,
        std_dev_file_name: str,
        ci_min_file_name: str,
        ci_max_file_name: str,
        hof: MultiObjectiveOptimizerResult,
        objectives: Sequence[PersonalObjectiveWithImportance],
        train_data: InputData,
        test_data: Optional[InputData] = None,
        train_name: str = "train", cv_name: str = INNER_CV_STR, test_name: str = TEST_STR):
    Path(path_saves).mkdir(parents=True, exist_ok=True)
    df_fit, df_sd, df_ci_min, df_ci_max = compute_all_fitnesses_with_confidence(
        hof=hof, objectives=objectives, test_data=train_data)  # Computing performance on train data here.
    df_fit = df_fit.add_prefix(str(train_name) + "_")
    df_sd = df_sd.add_prefix(str(train_name) + "_")
    df_ci_min = df_ci_min.add_prefix(str(train_name) + "_")
    df_ci_max = df_ci_max.add_prefix(str(train_name) + "_")
    if hof.has_fitnesses():
        inner_cv_fit_df = hof.fitnesses_to_df()
        inner_cv_fit_df.columns = [o.nick() for o in objectives]
        inner_cv_fit_df = inner_cv_fit_df.add_prefix(str(cv_name) + "_")
        df_fit = concat((inner_cv_fit_df, df_fit), axis=1)
        inner_cv_sd_df = hof.std_dev_to_df()
        inner_cv_sd_df.columns = [o.nick() for o in objectives]
        inner_cv_sd_df = inner_cv_sd_df.add_prefix(str(cv_name) + "_")
        df_sd = concat((inner_cv_sd_df, df_sd), axis=1)
        inner_cv_ci_min_df = hof.ci_min_to_df()
        inner_cv_ci_min_df.columns = [o.nick() for o in objectives]
        inner_cv_ci_min_df = inner_cv_ci_min_df.add_prefix(str(cv_name) + "_")
        df_ci_min = concat((inner_cv_ci_min_df, df_ci_min), axis=1)
        inner_cv_ci_max_df = hof.ci_max_to_df()
        inner_cv_ci_max_df.columns = [o.nick() for o in objectives]
        inner_cv_ci_max_df = inner_cv_ci_max_df.add_prefix(str(cv_name) + "_")
        df_ci_max = concat((inner_cv_ci_max_df, df_ci_max), axis=1)
    if test_data is not None:
        test_fit_df, test_sd_df, test_ci_min_df, test_ci_max_df = compute_all_fitnesses_with_confidence(
            hof=hof, objectives=objectives, test_data=test_data)
        test_fit_df = test_fit_df.add_prefix(str(test_name) + "_")
        test_sd_df = test_sd_df.add_prefix(str(test_name) + "_")
        test_ci_min_df = test_ci_min_df.add_prefix(str(test_name) + "_")
        test_ci_max_df = test_ci_max_df.add_prefix(str(test_name) + "_")
        df_fit = concat((df_fit, test_fit_df), axis=1)
        df_sd = concat((df_sd, test_sd_df), axis=1)
        df_ci_min = concat((df_ci_min, test_ci_min_df), axis=1)
        df_ci_max = concat((df_ci_max, test_ci_max_df), axis=1)
    df_fit.to_csv(path_or_buf=path_saves + fitnesses_file_name, index=False)
    df_sd.to_csv(path_or_buf=path_saves + std_dev_file_name, index=False)
    df_ci_min.to_csv(path_or_buf=path_saves + ci_min_file_name, index=False)
    df_ci_max.to_csv(path_or_buf=path_saves + ci_max_file_name, index=False)


def solution_features_file_name(fold_index: int) -> str:
    return SOLUTION_FEATURES_PREFIX + str(fold_index) + CSV_EXTENSION


def solution_fitnesses_file_name(fold_index: int) -> str:
    return SOLUTION_FITNESSES_FOLD_PREFIX + str(fold_index) + CSV_EXTENSION


def solution_std_devs_file_name(fold_index: int) -> str:
    return SOLUTION_STD_DEVS_FOLD_PREFIX + str(fold_index) + CSV_EXTENSION


def solution_ci_min_file_name(fold_index: int) -> str:
    return SOLUTION_CI_MIN_FOLD_PREFIX + str(fold_index) + CSV_EXTENSION


def solution_ci_max_file_name(fold_index: int) -> str:
    return SOLUTION_CI_MAX_FOLD_PREFIX + str(fold_index) + CSV_EXTENSION


class HofsSaver(MultiObjectiveCrossEvaluator):
    __save_path: str
    __objectives: Sequence[PersonalObjectiveWithImportance]

    def __init__(self, save_path: str, objectives: Sequence[PersonalObjectiveWithImportance]):
        self.__save_path = save_path
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry(),
                 n_proc: Optional[int] = None):
        n_folds = len(non_dominated_predictors_with_hyperparams)
        if n_folds > 0:
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
                train_data = input_data.select_samples(row_indices=folds.train_indices(fold))
                test_data = input_data.select_samples(row_indices=folds.test_indices(fold))
                save_hof_fitnesses_with_confidence(
                    path_saves=path_saves,
                    fitnesses_file_name=solution_fitnesses_file_name(fold_index=fold),
                    std_dev_file_name=solution_std_devs_file_name(fold_index=fold),
                    ci_min_file_name=solution_ci_min_file_name(fold_index=fold),
                    ci_max_file_name=solution_ci_max_file_name(fold_index=fold),
                    hof=hof,
                    objectives=self.__objectives,
                    train_data=train_data,
                    test_data=test_data)

    def name(self) -> str:
        return "Hall of fames saver"
