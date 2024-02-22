import numpy
from numpy.random import shuffle
from pandas import DataFrame

from model.model import Predictor
from objective.objective_computer import ObjectiveComputer
from util.dataframes import n_col, n_row, replace_column


def feature_importance_by_permutation(
        objective_computer: ObjectiveComputer, predictor: Predictor,
        x_test: DataFrame, y_test: DataFrame, seed: int = 764254) -> list[float]:
    """This function modifies x_test in place to save memory and returns it in its original state afterwards."""
    performance_full = objective_computer.compute_from_predictor_and_test(
        predictor=predictor, x_test=x_test, y_test=y_test).fitness()
    generator = numpy.random.default_rng(seed=seed)
    n_features = n_col(x_test)
    n_samples = n_row(x_test)
    shuffling_idx = numpy.arange(n_samples)
    generator.shuffle(shuffling_idx)
    importances = [0.0] * n_features
    for i in range(n_features):
        old_col = list(x_test.iloc[:, i])
        # Must use list to make sure that we get an actual copy and not a shallow one.
        temp_col = list(x_test.iloc[shuffling_idx, i])
        # Must use list to ignore previous indices that are still in Series.
        replace_column(df=x_test, col_pos=i, col_data=temp_col)
        performance_i = objective_computer.compute_from_predictor_and_test(
            predictor=predictor, x_test=x_test, y_test=y_test).fitness()
        importances[i] = max(performance_full - performance_i, 0.0)
        replace_column(df=x_test, col_pos=i, col_data=old_col)
    return importances
