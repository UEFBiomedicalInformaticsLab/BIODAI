from collections.abc import Sequence

import numpy as np
from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from evaluator.evaluate_individual import fit_inner_model
from individual.mv_feature_set_by_names import MVFeatureSetByNames
from input_data.evaluation_ready_input_data import NoOutcomesInputData
from input_data.input_data import InputData, select_common_features
from input_data.model_ready_input_data import ModelReadyInputData
from input_data.outcome import smart_create_outcome
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import OutPrinter, Printer
from util.randoms import set_all_seeds
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.table.table_utils import n_col
from views.views import JustViews

DEFAULT_FRACTIONS = (1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/32.0, 1.0/64.0, 0.0)
DEFAULT_SIGMAS = (10.0, 5.0, 2.0, 1.0, 0.5, 0.0)


def shuffle_features_df(data: DataFrame, fraction: float) -> DataFrame:
    """ Randomly shuffle feature values within columns based on a given fraction.
    Does not modify original dataframe."""
    data = data.copy()
    for column in data.columns:
        n = len(data)
        for i in range(n):
            if np.random.rand() < fraction:
                swap_idx = np.random.randint(0, n)
                # Swap the current value with another random value in the same column
                data.at[i, column], data.at[swap_idx, column] = data.at[swap_idx, column], data.at[i, column]
    return data


def add_gaussian_noise_df(data: DataFrame, sigma: float) -> DataFrame:
    """ Add Gaussian noise to feature values. Does not modify original dataframe."""
    data = data.copy()
    noise = np.random.normal(0, sigma, data.shape)
    features = data.columns
    data[features] += noise
    return data


def shuffle_features_table(table: Table, fraction: float) -> Table:
    df = table.to_dataframe()
    shuffled_df = shuffle_features_df(data=df, fraction=fraction)
    return BackedTable(NpTable(data=shuffled_df))


def add_gaussian_noise_table(table: Table, sigma: float) -> Table:
    """ Add Gaussian noise to feature values."""
    df = table.to_dataframe()
    noisy_df = add_gaussian_noise_df(data=df, sigma=sigma)
    return BackedTable(NpTable(data=noisy_df))


def shuffle_features_input_data(data: InputData, fraction: float) -> InputData:
    views = data.views()
    view_names = views.keys()
    shuffled_tables = {}
    for name in view_names:
        shuffled_tables[name] = shuffle_features_table(table=views.view(key=name), fraction=fraction)
    new_views = JustViews(views_dict=shuffled_tables)
    return data.set_views(views=new_views)


def add_gaussian_noise_input_data(data: InputData, sigma: float) -> InputData:
    views = data.views()
    view_names = views.keys()
    noisy_tables = {}
    for name in view_names:
        noisy_tables[name] = add_gaussian_noise_table(table=views.view(key=name), sigma=sigma)
    new_views = JustViews(views_dict=noisy_tables)
    return data.set_views(views=new_views)


def ablation_study_altered_data(
        train_x_df: DataFrame,
        train_outcomes: dict[str, DataFrame],
        test_x_df: DataFrame,
        test_outcomes: dict[str, DataFrame],
        objectives: Sequence[PersonalObjectiveWithImportance],
        seed: int = 83756) -> Sequence[CVResult]:
    set_all_seeds(seed=seed)
    res = []
    individual = [True]*n_col(train_x_df)
    train_views = JustViews.create_from_dfs(views_dict={"train x": train_x_df})
    test_views = JustViews.create_from_dfs(views_dict={"test x": test_x_df})
    for objective in objectives:
        if objective.requires_predictions():
            outcome_label = objective.outcome_label()
            train_outcome = smart_create_outcome(y=train_outcomes[outcome_label], name="train outcome")
            train_data = ModelReadyInputData(all_views=train_views, outcome=train_outcome, nick="train")
            test_outcome = smart_create_outcome(y=test_outcomes[outcome_label], name="test outcome")
            test_data = ModelReadyInputData(all_views=test_views, outcome=test_outcome, nick="test")
            predictor = fit_inner_model(
                train_filtered_data = train_data,
                model=objective.mv_model())
            cv_res = objective.compute_from_predictor_and_test_mv(
                predictor=predictor,
                test_data=test_data,
                train_data=train_data)
            res.append(cv_res)
        else:
            if objective.has_outcome_label():
                outcome_label = objective.outcome_label()
                test_outcome = smart_create_outcome(y=test_outcomes[outcome_label], name="test outcome")
                test_data = ModelReadyInputData(all_views=test_views, outcome=test_outcome, nick="test")
            else:
                test_data = NoOutcomesInputData(all_views=test_views, nick="test")
            objective_computer = objective.objective_computer()
            if objective_computer.requires_target():
                cv_res = objective_computer.compute_from_structure_with_importance(
                    hyperparams=individual, data=test_data,
                    compute_fi=False, compute_confidence=True)
            else:
                cv_res = objective.compute_from_hyperparams_all(hyperparams_seq=individual)
            res.append(cv_res)
    return res


def ablation_study_fraction(
        train_collapsed_views: DataFrame,
        train_outcomes: dict[str, DataFrame],
        test_collapsed_views: DataFrame,
        test_outcomes: dict[str, DataFrame],
        objectives: Sequence[PersonalObjectiveWithImportance],
        fraction: float) -> Sequence[CVResult]:
    train_x_df = shuffle_features_df(data=train_collapsed_views, fraction=fraction)
    test_x_df = shuffle_features_df(data=test_collapsed_views, fraction=fraction)
    return ablation_study_altered_data(
        train_x_df=train_x_df,
        train_outcomes=train_outcomes,
        test_x_df=test_x_df,
        test_outcomes=test_outcomes,
        objectives=objectives)


def ablation_study_sigma(
        train_collapsed_views: DataFrame,
        train_outcomes: dict[str, DataFrame],
        test_collapsed_views: DataFrame,
        test_outcomes: dict[str, DataFrame],
        objectives: Sequence[PersonalObjectiveWithImportance],
        sigma: float) -> Sequence[CVResult]:
    train_x_df = add_gaussian_noise_df(data=train_collapsed_views, sigma=sigma)
    test_x_df = add_gaussian_noise_df(data=test_collapsed_views, sigma=sigma)
    return ablation_study_altered_data(
        train_x_df=train_x_df,
        train_outcomes=train_outcomes,
        test_x_df=test_x_df,
        test_outcomes=test_outcomes,
        objectives=objectives)


def prepare_data_for_ablation(
        train_data: InputData, test_data: InputData,
        feature_names: MVFeatureSetByNames,
        printer: Printer = OutPrinter()) -> tuple[DataFrame, DataFrame]:
    printer.title_print("Processing ablation study for feature set " + feature_names.name())
    printer.print("Number of features in set: " + str(feature_names.n_features()))
    printer.title_print("Selecting features specified for ablation study.")
    train_data = train_data.select_existing_features(features=feature_names)
    test_data = test_data.select_existing_features(features=feature_names)
    printer.title_print("Reducing datasets to common features.")
    train_data, test_data = select_common_features(a=train_data, b=test_data)
    printer.print("Number of features present in both datasets: " + str(train_data.n_features()))
    printer.title_print("Standardizing the features of each dataset separately.")
    train_data = train_data.standardize_features()
    test_data = test_data.standardize_features()
    printer.print("Train data")
    printer.print(train_data)
    printer.print("Test data")
    printer.print(test_data)
    train_collapsed_views = train_data.collapsed_views().to_dataframe()
    test_collapsed_views = test_data.collapsed_views().to_dataframe()
    return train_collapsed_views, test_collapsed_views


def ablation_study_fractions(
        train_data: InputData, test_data: InputData,
        objectives: Sequence[PersonalObjectiveWithImportance],
        feature_names: MVFeatureSetByNames,
        fractions: Sequence[float] = DEFAULT_FRACTIONS,
        printer: Printer = OutPrinter()) -> Sequence[Sequence[CVResult]]:
    """One element for each fraction. Each element has an element for each objective."""
    train_collapsed_views, test_collapsed_views = prepare_data_for_ablation(
        train_data=train_data,
        test_data=test_data,
        feature_names=feature_names,
        printer=printer)
    res = []
    for fraction in fractions:
        res.append(ablation_study_fraction(
            train_collapsed_views=train_collapsed_views,
            train_outcomes=train_data.outcomes_data_dict(),
            test_collapsed_views=test_collapsed_views,
            test_outcomes=test_data.outcomes_data_dict(),
            objectives=objectives,
            fraction=fraction))
    return res


def ablation_study_sigmas(
        train_data: InputData, test_data: InputData,
        objectives: Sequence[PersonalObjectiveWithImportance],
        feature_names: MVFeatureSetByNames,
        sigmas: Sequence[float] = DEFAULT_SIGMAS,
        printer: Printer = OutPrinter()) -> Sequence[Sequence[CVResult]]:
    """One element for each sigma. Each element has an element for each objective."""
    train_collapsed_views, test_collapsed_views = prepare_data_for_ablation(
        train_data=train_data,
        test_data=test_data,
        feature_names=feature_names,
        printer=printer)
    res = []
    for sigma in sigmas:
        res.append(ablation_study_sigma(
            train_collapsed_views=train_collapsed_views,
            train_outcomes=train_data.outcomes_data_dict(),
            test_collapsed_views=test_collapsed_views,
            test_outcomes=test_data.outcomes_data_dict(),
            objectives=objectives,
            sigma=sigma))
    return res


def ablation_results_to_lines(results: Sequence[Sequence[CVResult]]) -> Sequence[Sequence[float]]:
    if len(results) == 0:
        return []
    n_lines = len(results[0])
    return [[r[m].fitness() for r in results] for m in range(n_lines)]
