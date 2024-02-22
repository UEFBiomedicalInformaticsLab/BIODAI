from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from pandas import DataFrame

from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.model import Predictor
from objective.objective_computer import ObjectiveComputer
from util.dataframes import n_row, select_by_row_indices
from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.math.online_variance_builder import OnlineVarianceBuilder
from util.sequence_utils import select_by_indices, list_of_empty_lists
from views.views import Views


def distribution_to_result(distribution: Sequence[float]) -> tuple[Interval, float, float]:
    interval_min = np.percentile(distribution, 2.5, interpolation='linear')
    interval_max = np.percentile(distribution, 97.5, interpolation='linear')
    var_builder = OnlineVarianceBuilder()
    var_builder.add_all(elems=distribution)
    return ConcreteInterval(a=interval_min, b=interval_max), var_builder.sample_standard_deviation(), var_builder.mean()


def create_resample_views(x_test: Views, y_test: DataFrame) -> tuple[Views, DataFrame]:
    """Uses numpy.random."""
    size = x_test.n_samples()
    locs = np.random.choice(a=size, size=size, replace=True)
    res_y = y_test.take(locs, axis=0)
    res_x = x_test.select_samples(locs=locs)
    return res_x, res_y


def create_resample_dataframe(x_test: DataFrame, y_test: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Uses numpy.random."""
    size = n_row(x_test)
    locs = np.random.choice(a=size, size=size, replace=True)
    res_y = y_test.take(locs, axis=0)
    return x_test.take(locs, axis=0), res_y


def create_resample_dataframe_or_views(
        x_test: Union[DataFrame, Views], y_test: DataFrame,
        cache_concatenation: bool = False) -> tuple[Union[DataFrame, Views], DataFrame]:
    """Uses numpy.random."""
    if isinstance(x_test, Views):
        x_resampled, y_resampled = create_resample_views(x_test=x_test, y_test=y_test)
        if cache_concatenation:
            x_resampled = x_resampled.as_cached()
        return x_resampled, y_resampled
    elif isinstance(x_test, DataFrame):
        return create_resample_dataframe(x_test=x_test, y_test=y_test)
    else:
        raise ValueError("Unexpected x_test type: " + str(type(x_test)))


def bootstrap_one_resample_from_classes(
        objective_computer: ObjectiveComputer, pred_y_test: Sequence, true_y_test: Sequence,
        pred_y_train: Optional[Sequence], true_y_train: Optional[Sequence]) -> float:
    """Uses numpy.random. Only the test labels will be resampled."""
    size = len(pred_y_test)
    locs = np.random.choice(a=size, size=size, replace=True)
    if isinstance(pred_y_test, DataFrame):
        pred_resampled = select_by_row_indices(samples=pred_y_test, indices=locs)
    else:
        pred_resampled = select_by_indices(data=pred_y_test, indices=locs)
    if isinstance(true_y_test, DataFrame):
        true_resampled = select_by_row_indices(samples=true_y_test, indices=locs)
    else:
        true_resampled = select_by_indices(data=true_y_test, indices=locs)
    return objective_computer.compute_from_classes(
        hyperparams=None, hp_manager=None,
        test_pred=pred_resampled, test_true=true_resampled, train_pred=pred_y_train, train_true=true_y_train).fitness()


def bootstrap_one_resample_with_predictor(
        objective_computer: ObjectiveComputer, predictor: Predictor,
        x_test: Union[DataFrame, Views], y_test: DataFrame) -> float:
    """Uses numpy.random."""
    x_resampled, y_resampled = create_resample_dataframe_or_views(x_test=x_test, y_test=y_test)
    return objective_computer.compute_from_predictor_and_test(
        predictor=predictor, x_test=x_resampled, y_test=y_resampled).fitness()


def bootstrap_one_resample_with_predictors(
        objective_computer: ObjectiveComputer, predictors: Sequence[Predictor],
        x_test: Union[DataFrame, Views], y_test: DataFrame) -> list[float]:
    """Uses numpy.random."""
    x_resampled, y_resampled = create_resample_dataframe_or_views(
        x_test=x_test, y_test=y_test, cache_concatenation=True)
    return [objective_computer.compute_from_predictor_and_test(
        predictor=p, x_test=x_resampled, y_test=y_resampled).fitness() for p in predictors]


def bootstrap_distribution_from_classes(
        objective_computer: ObjectiveComputer,
        pred_y_test: Sequence, true_y_test: Sequence, n_resamples: int,
        pred_y_train: Optional[Sequence], true_y_train: Optional[Sequence]) -> list[float]:
    """Uses numpy.random.
       Returns the fitnesses in ascending order.
       Only the test labels will be resampled."""
    res = []
    for _ in range(n_resamples):
        res.append(bootstrap_one_resample_from_classes(
            objective_computer=objective_computer,
            pred_y_test=pred_y_test, true_y_test=true_y_test,
            pred_y_train=pred_y_train, true_y_train=true_y_train))
        res.sort()
    return res


def bootstrap_distribution(objective_computer: ObjectiveComputer, predictor: Predictor,
                           x_test: DataFrame, y_test: DataFrame, n_resamples: int) -> list[float]:
    """Uses numpy.random.
       Returns the fitnesses in ascending order.
       TODO: can be faster when objective_computer can compute from classes."""
    res = []
    for _ in range(n_resamples):
        res.append(bootstrap_one_resample_with_predictor(
            objective_computer=objective_computer, predictor=predictor, x_test=x_test, y_test=y_test))
        res.sort()
    return res


def bootstrap_distribution_all(
        objective_computer: ObjectiveComputer, predictors: Sequence[Predictor],
        x_test: Union[DataFrame, Views], y_test: DataFrame, n_resamples: int) -> list[list[float]]:
    """Uses numpy.random.
       Returns for each predictor the fitnesses in ascending order.
       TODO: can be faster when objective_computer can compute from classes."""
    n_predictors = len(predictors)
    res = list_of_empty_lists(n=n_predictors)
    for _ in range(n_resamples):
        sample_res = bootstrap_one_resample_with_predictors(
            objective_computer=objective_computer, predictors=predictors, x_test=x_test, y_test=y_test)
        for i in range(n_predictors):
            try:
                res[i].append(sample_res[i])
            except IndexError as e:
                raise IndexError(str(e) +
                                 "\nIndex: " + str(i) +
                                 "\nn_predictors: " + str(n_predictors) +
                                 "\nlen res: " + str(len(res)) +
                                 "\nlen sample_res: " + str(len(sample_res)))
    for i in range(n_predictors):
        res[i].sort()
    return res


def bootstrap_ci95_from_classes(
        objective_computer: ObjectiveComputer,
        pred_y_test: Sequence, true_y_test: Sequence,
        n_resamples: int, pred_y_train: Optional[Sequence], true_y_train: Optional[Sequence]) -> tuple[Interval, float, float]:
    """Uses numpy.random.
    Returns interval, standard deviation and bootstrap mean.
    Only the test labels will be resampled."""
    distribution = bootstrap_distribution_from_classes(
        objective_computer=objective_computer,
        pred_y_test=pred_y_test, true_y_test=true_y_test, n_resamples=n_resamples,
        pred_y_train=pred_y_train, true_y_train=true_y_train)
    return distribution_to_result(distribution=distribution)


def bootstrap_ci95(objective_computer: ObjectiveComputer, predictor: Predictor,
                   x_test: DataFrame, y_test: DataFrame, n_resamples: int) -> tuple[Interval, float, float]:
    """Uses numpy.random.
    Returns interval, standard deviation and bootstrap mean."""
    distribution = bootstrap_distribution(objective_computer=objective_computer, predictor=predictor,
                                          x_test=x_test, y_test=y_test, n_resamples=n_resamples)
    return distribution_to_result(distribution=distribution)


def bootstrap_ci95_all(
        objective_computer: ObjectiveComputer, predictors: Sequence[Predictor],
        x_test: DataFrame, y_test: DataFrame, n_resamples: int) -> list[tuple[Interval, float, float]]:
    """Uses numpy.random.
    Returns interval, standard deviation and bootstrap mean."""
    distributions = bootstrap_distribution_all(
        objective_computer=objective_computer, predictors=predictors,
        x_test=x_test, y_test=y_test, n_resamples=n_resamples)
    return [distribution_to_result(distribution=d) for d in distributions]


def bootstrap_one_resample_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Union[HyperparamManager, None],
        x_test: Union[DataFrame, Views], y_test: DataFrame) -> float:
    """x_test can be DataFrame or Views.
    Uses numpy.random."""
    x_resampled, y_resampled = create_resample_dataframe_or_views(x_test=x_test, y_test=y_test)
    return objective_computer.compute_from_structure(
        hyperparams=hyperparams, hp_manager=hp_manager,
        x=x_resampled, y=y_resampled).fitness()


def bootstrap_distribution_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Union[HyperparamManager, None],
        x_test: Union[DataFrame, Views], y_test: DataFrame, n_resamples: int) -> list[float]:
    """Uses numpy.random.
       Returns the fitnesses in ascending order."""
    res = []
    for _ in range(n_resamples):
        res.append(bootstrap_one_resample_from_structure(
            objective_computer=objective_computer,
            hyperparams=hyperparams, hp_manager=hp_manager,
            x_test=x_test, y_test=y_test))
        res.sort()
    return res


def bootstrap_ci95_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Union[HyperparamManager, None],
        x_test: DataFrame, y_test: DataFrame, n_resamples: int) -> tuple[Interval, float, float]:
    """Uses numpy.random.
        Returns interval, standard deviation and bootstrap mean."""
    distribution = bootstrap_distribution_from_structure(
        objective_computer=objective_computer,
        hyperparams=hyperparams, hp_manager=hp_manager,
        x_test=x_test, y_test=y_test, n_resamples=n_resamples)
    return distribution_to_result(distribution=distribution)
