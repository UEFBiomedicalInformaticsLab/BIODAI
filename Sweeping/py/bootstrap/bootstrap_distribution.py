from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from numpy.random import Generator, default_rng
from pandas import DataFrame

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from model.sv_model import SVPredictor
from objective.objective_computer import ObjectiveComputer
from util.dataframe.dataframes import select_by_row_indices
from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.math.online_variance_builder import OnlineVarianceBuilder
from util.sequence_utils import list_of_empty_lists
from util.select_from_sequence import select_by_indices
from views.views import Views


DEFAULT_RESAMPLING_SEED = 34624


def distribution_to_result(distribution: Sequence[float]) -> tuple[Interval, float, float]:
    try:
        interval_min = np.percentile(distribution, 2.5, interpolation='linear')
        interval_max = np.percentile(distribution, 97.5, interpolation='linear')
    except IndexError as e:
        print("Index error in distribution_to_result.")
        print("Distribution:\n" + str(distribution))
        raise e
    var_builder = OnlineVarianceBuilder()
    var_builder.add_all(elems=distribution)
    return ConcreteInterval(a=interval_min, b=interval_max), var_builder.sample_standard_deviation(), var_builder.mean()


def create_resample_views(x_test: Views, y_test: DataFrame, random: Generator) -> tuple[Views, DataFrame]:
    size = x_test.n_samples()
    locs = random.choice(a=size, size=size, replace=True)
    res_y = y_test.take(locs, axis=0)
    res_x = x_test.select_samples(locs=locs)
    return res_x, res_y


def create_resample_dataframe(x_test: DataFrame, y_test: DataFrame, random: Generator) -> tuple[DataFrame, DataFrame]:
    """Uses numpy.random."""
    from util.table.table_utils import n_row
    size = n_row(x_test)
    locs = random.choice(a=size, size=size, replace=True)
    res_y = y_test.take(locs, axis=0)
    return x_test.take(locs, axis=0), res_y


def create_resample_dataframe_or_views(
        x_test: Union[DataFrame, Views], y_test: DataFrame,
        random: Generator,
        cache_concatenation: bool = False) -> tuple[Union[DataFrame, Views], DataFrame]:
    """If cache_concatenation is true and x_test is Views, returns views that cache the concatenation."""
    if isinstance(x_test, Views):
        x_resampled, y_resampled = create_resample_views(x_test=x_test, y_test=y_test, random=random)
        if cache_concatenation:
            x_resampled = x_resampled.as_cached()
        return x_resampled, y_resampled
    elif isinstance(x_test, DataFrame):
        return create_resample_dataframe(x_test=x_test, y_test=y_test, random=random)
    else:
        raise ValueError("Unexpected x_test type: " + str(type(x_test)))


def create_resample_input_data(
        data: EvaluationReadyInputData,
        random: Generator) -> EvaluationReadyInputData:
    size = data.n_samples()
    locs = random.choice(a=size, size=size, replace=True)
    return data.select_samples(row_indices=locs)


def create_resample_model_input_data(
        data: ModelReadyInputData,
        random: Generator) -> ModelReadyInputData:
    size = data.n_samples()
    locs = random.choice(a=size, size=size, replace=True)
    return data.select_samples(row_indices=locs)


def bootstrap_one_resample_from_classes(
        objective_computer: ObjectiveComputer, pred_y_test: Sequence, true_y_test: Sequence,
        pred_y_train: Optional[Sequence], true_y_train: Optional[Sequence], random: Generator) -> float:
    """Uses a random Generator. Only the test labels will be resampled."""
    size = len(pred_y_test)
    locs = random.choice(a=np.arange(size), size=size, replace=True)
    if isinstance(pred_y_test, DataFrame):
        pred_resampled = select_by_row_indices(samples=pred_y_test, indices=locs)
    else:
        pred_resampled = select_by_indices(data=pred_y_test, indices=locs)
    if isinstance(true_y_test, DataFrame):
        true_resampled = select_by_row_indices(samples=true_y_test, indices=locs)
    else:
        true_resampled = select_by_indices(data=true_y_test, indices=locs)
    return objective_computer.compute_from_classes_mv(
        test_pred=pred_resampled, test_true=true_resampled, train_pred=pred_y_train, train_true=true_y_train,
        hyperparams=None, hp_manager=None).fitness()


def bootstrap_one_resample_with_predictor_mv(
        objective_computer: ObjectiveComputer, predictor: MVPredictor,
        test_data: ModelReadyInputData, random: Generator) -> float:
    resampled = create_resample_model_input_data(data=test_data, random=random)
    return objective_computer.compute_from_predictor_and_test_mv(
        predictor=predictor, test_data=resampled).fitness()


def bootstrap_one_resample_with_predictors_mv(
        objective_computer: ObjectiveComputer, predictors: Sequence[MVPredictor],
        test_data: ModelReadyInputData, random: Generator) -> list[float]:
    resampled = create_resample_model_input_data(
        data=test_data, random=random)
    return [objective_computer.compute_from_predictor_and_test_mv(
        predictor=p, test_data=resampled).fitness() for p in predictors]


def bootstrap_distribution_from_classes(
        objective_computer: ObjectiveComputer,
        pred_y_test: Sequence, true_y_test: Sequence, n_resamples: int,
        random: Generator,
        pred_y_train: Optional[Sequence] = None, true_y_train: Optional[Sequence] = None) -> list[float]:
    """System randoms are not touched.
       Returns the fitnesses in ascending order.
       Only the test labels will be resampled."""
    res = [bootstrap_one_resample_from_classes(
            objective_computer=objective_computer,
            pred_y_test=pred_y_test, true_y_test=true_y_test,
            pred_y_train=pred_y_train, true_y_train=true_y_train,
            random=random) for _ in range(n_resamples)]
    res.sort()
    return res


def bootstrap_distribution_class(
        objective_computer: ObjectiveComputer, predictor: SVPredictor,
        x_test: DataFrame, y_test: DataFrame, n_resamples: int,
        seed: int = DEFAULT_RESAMPLING_SEED) -> list[float]:
    """Returns the fitnesses in ascending order."""
    assert objective_computer.is_class_objective_computer()
    y_pred = predictor.predict(x=x_test)
    return bootstrap_distribution_from_classes(
        objective_computer=objective_computer,
        pred_y_test=y_pred, true_y_test=y_test, n_resamples=n_resamples, random=np.random.default_rng(seed=seed))


def bootstrap_distribution_mv_class(
        objective_computer: ObjectiveComputer, predictor: MVPredictor,
        test_data: ModelReadyInputData, n_resamples: int, seed: int = DEFAULT_RESAMPLING_SEED) -> list[float]:
    """Does not use system randoms.
       Returns the fitnesses in ascending order.
       """
    assert objective_computer.is_class_objective_computer()
    y_pred = predictor.predict(views=test_data.views())
    return bootstrap_distribution_from_classes(
        objective_computer=objective_computer,
        pred_y_test=y_pred, true_y_test=test_data.outcome_data(),
        n_resamples=n_resamples, random=np.random.default_rng(seed=seed))


def bootstrap_distribution_mv(
        objective_computer: ObjectiveComputer, predictor: MVPredictor,
        test_data: ModelReadyInputData, n_resamples: int, seed: int = DEFAULT_RESAMPLING_SEED) -> list[float]:
    """Returns the fitnesses in ascending order.
    Consider to pass input data with cached and compiled collapsed table for best performance."""
    if objective_computer.can_compute_from_classes():
        return bootstrap_distribution_mv_class(
            objective_computer=objective_computer,  predictor=predictor,test_data=test_data, n_resamples=n_resamples,
            seed=seed)
    else:
        res = []
        random = default_rng(seed=seed)
        for _ in range(n_resamples):
            res.append(bootstrap_one_resample_with_predictor_mv(
                objective_computer=objective_computer, predictor=predictor, test_data=test_data,
                random=random))
        res.sort()
        return res


def bootstrap_distribution_all_mv_class(
        objective_computer: ObjectiveComputer, predictors: Sequence[MVPredictor],
        test_data: ModelReadyInputData, n_resamples: int, seed: int = DEFAULT_RESAMPLING_SEED) -> list[list[float]]:
    """Does not use system randoms.
       Returns for each predictor the fitnesses in ascending order.
       All predictors are tested on the same resamples.
       """
    assert objective_computer.is_class_objective_computer()
    res = []
    for p in predictors:
        y_pred = p.predict(views=test_data.views())
        res.append(bootstrap_distribution_from_classes(
            objective_computer=objective_computer,
            pred_y_test=y_pred, true_y_test=test_data.outcome_data(),
            n_resamples=n_resamples, random=np.random.default_rng(seed=seed)))
    return res


def bootstrap_distribution_all_mv(
        objective_computer: ObjectiveComputer, predictors: Sequence[MVPredictor],
        test_data: ModelReadyInputData, n_resamples: int, seed: int = DEFAULT_RESAMPLING_SEED) -> list[list[float]]:
    """Does not touch system randoms. Returns for each predictor the fitnesses in ascending order."""
    if objective_computer.is_class_objective_computer():
        return bootstrap_distribution_all_mv_class(
            objective_computer=objective_computer, predictors=predictors,
            test_data=test_data, n_resamples=n_resamples, seed=seed)
    else:
        n_predictors = len(predictors)
        res = list_of_empty_lists(n=n_predictors)
        random = np.random.default_rng(seed=seed)
        for _ in range(n_resamples):
            sample_res = bootstrap_one_resample_with_predictors_mv(
                objective_computer=objective_computer, predictors=predictors, test_data=test_data,
                random=random)
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
        n_resamples: int, pred_y_train: Optional[Sequence], true_y_train: Optional[Sequence],
        random: Generator) -> tuple[Interval, float, float]:
    """System randoms are not touched.
    Returns interval, standard deviation and bootstrap mean.
    Only the test labels will be resampled."""
    distribution = bootstrap_distribution_from_classes(
        objective_computer=objective_computer,
        pred_y_test=pred_y_test, true_y_test=true_y_test, n_resamples=n_resamples,
        pred_y_train=pred_y_train, true_y_train=true_y_train, random=random)
    return distribution_to_result(distribution=distribution)


def bootstrap_ci95_mv(
        objective_computer: ObjectiveComputer, predictor: MVPredictor,
        test_data: ModelReadyInputData, n_resamples: int) -> tuple[Interval, float, float]:
    """Uses numpy.random.
    Returns interval, standard deviation and bootstrap mean."""
    distribution = bootstrap_distribution_mv(
        objective_computer=objective_computer, predictor=predictor,
        test_data=test_data, n_resamples=n_resamples)
    return distribution_to_result(distribution=distribution)


def bootstrap_ci95_all_mv(
        objective_computer: ObjectiveComputer, predictors: Sequence[MVPredictor],
        test_data: ModelReadyInputData, n_resamples: int,
        seed: int = DEFAULT_RESAMPLING_SEED) -> list[tuple[Interval, float, float]]:
    """Returns interval, standard deviation and bootstrap mean.
    Returns a result for each predictor."""
    distributions = bootstrap_distribution_all_mv(
        objective_computer=objective_computer, predictors=predictors,
        test_data=test_data, n_resamples=n_resamples, seed=seed)
    return [distribution_to_result(distribution=d) for d in distributions]


def bootstrap_one_resample_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Optional[MvHyperparamManager],
        test_data: EvaluationReadyInputData, random: Generator) -> float:
    resampled = create_resample_input_data(data=test_data, random=random)
    return objective_computer.compute_from_structure(
        hyperparams=hyperparams, hp_manager=hp_manager, data=resampled).fitness()


def bootstrap_distribution_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Optional[MvHyperparamManager],
        test_data: EvaluationReadyInputData, n_resamples: int,
        seed: int = DEFAULT_RESAMPLING_SEED) -> list[float]:
    """Uses numpy.random.
       Returns the fitnesses in ascending order."""
    res = []
    random = np.random.default_rng(seed=seed)
    for _ in range(n_resamples):
        res.append(bootstrap_one_resample_from_structure(
            objective_computer=objective_computer,
            hyperparams=hyperparams, hp_manager=hp_manager,
            test_data=test_data, random=random))
        res.sort()
    return res


def bootstrap_ci95_from_structure(
        objective_computer: ObjectiveComputer,
        hyperparams, hp_manager: Optional[MvHyperparamManager],
        test_data: EvaluationReadyInputData, n_resamples: int) -> tuple[Interval, float, float]:
    """Uses numpy.random.
        Returns interval, standard deviation and bootstrap mean."""
    distribution = bootstrap_distribution_from_structure(
        objective_computer=objective_computer,
        hyperparams=hyperparams, hp_manager=hp_manager,
        test_data=test_data, n_resamples=n_resamples)
    return distribution_to_result(distribution=distribution)
