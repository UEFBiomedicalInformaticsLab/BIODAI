import multiprocessing
from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.single_objective.all_fold_fitnesses import AllFitnesses
from cross_validation.single_objective.cv_result import CVResult
from cross_validation.validate_one_predictor_res import ValidateOnePredictorRes
from cross_validation.validate_predictor_worker_state import ValidateOnePredictorWorkerState, \
    predictor_validation_parallel_init, predictor_multiprocessing_friendly_validation
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.confident_individual import get_ci95s, get_std_devs, get_bootstrap_means
from individual.fit_individual import get_fitnesses
from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from input_data.input_data import InputData
from input_data.model_ready_input_data import ModelReadyInputData
from model.class_crisp.sv_classifier import ClassSVModel
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from objective.objective_with_importance.social_objective_with_importance import SocialObjectiveWithImportance
from prediction_stats.stat_creator import StatCreator
from util.dataframe.dataframes import select_by_row_indices
from util.randoms import random_seed


# Each fold is a pair training-testing
def select_all_sets(x, y, fold):
    train_indices = fold[0]
    test_indices = fold[1]
    x_train = select_by_row_indices(x, train_indices)
    y_train = select_by_row_indices(y, train_indices)
    x_test = select_by_row_indices(x, test_indices)
    y_test = select_by_row_indices(y, test_indices)
    return x_train, y_train, x_test, y_test


# x is a list of anything, each element being a sample
# y is a list of anything, each element being an expected output
def cross_validate(x, y, folds_list, model: ClassSVModel):
    test_pred_y = []
    test_true_y = []
    train_pred_y = []
    train_true_y = []
    for fold in folds_list:
        x_train, y_train, x_test, y_test = select_all_sets(x=x, y=y, fold=fold)
        predictions_on_train, predictions_on_test = model.fit_and_predict(
            x_train=x_train, y_train=y_train, x_test=x_test)
        train_pred_y.append(predictions_on_train)
        train_true_y.append(y_train)
        test_pred_y.append(predictions_on_test)
        test_true_y.append(y_test)
    return train_pred_y, train_true_y, test_pred_y, test_true_y


# x is a list of anything, each element being a sample
# y is a list of anything, each element being an expected output
def cross_validate_and_create_stats(x, y, folds_list, model: ClassSVModel, stat_creator: StatCreator):
    train_pred_y, train_true_y, test_pred_y, test_true_y = cross_validate(x=x, y=y, folds_list=folds_list, model=model)
    return stat_creator.create_stats(
        test_predicted_y=test_pred_y, test_true_y=test_true_y,
        train_predicted_y=train_pred_y, train_true_y=train_true_y)


def validate_single_sample_and_structural_objective(
        train_data: EvaluationReadyInputData, test_data: EvaluationReadyInputData,
        hyperparams, hp_manager: MvHyperparamManager, objective_computer: ObjectiveComputerWithImportance) -> tuple[CVResult, CVResult]:
    computed_for_test = objective_computer.compute_from_structure_with_importance(
        hyperparams=hyperparams, hp_manager=hp_manager,
        data=hp_manager.filter_evaluation_ready_data(hyperparams=hyperparams, data=test_data),
        compute_confidence=True, compute_fi=False)
    computed_for_train = objective_computer.compute_from_structure_with_importance(
        hyperparams=hyperparams, hp_manager=hp_manager,
        data=hp_manager.filter_evaluation_ready_data(hyperparams=hyperparams, data=train_data),
        compute_confidence=True, compute_fi=False)
    return computed_for_test, computed_for_train


def validate_one_predictor(
        train_data: EvaluationReadyInputData, test_data: EvaluationReadyInputData,
        predictor: MVPredictor, hyperparams, objective: SocialObjectiveWithImportance,
        compute_confidence: bool) -> ValidateOnePredictorRes:
    hp_manager = MaskMvHpManager.create_from_input_data(input_data=train_data)
    res = ValidateOnePredictorRes()
    if objective.is_class_based():
        if objective.requires_predictions():
            pred_train = predictor.predict(train_data.views())
            pred_test = predictor.predict(test_data.views())
            y_train = train_data.the_outcome().data()
            y_test = test_data.the_outcome().data()
            computed_for_test = objective.compute_from_classes_with_confidence(
                hyperparams=hyperparams, hp_manager=hp_manager, test_pred=pred_test,
                test_true=y_test, train_pred=pred_train,
                train_true=y_train, compute_confidence=compute_confidence)
            res.objective_on_test = computed_for_test.fitness()
            if compute_confidence:
                res.objective_on_test_ci = computed_for_test.ci95()
            if not objective.requires_training_predictions():
                computed_for_train = objective.compute_from_classes_with_confidence(
                    hyperparams=hyperparams, hp_manager=hp_manager,
                    test_pred=pred_train, test_true=y_train,
                    train_pred=None, train_true=None, compute_confidence=compute_confidence)
                res.objective_on_train = computed_for_train.fitness()
                if compute_confidence:
                    res.objective_on_train_ci = computed_for_train.ci95()
        else:
            computed_for_test, computed_for_train = validate_single_sample_and_structural_objective(
                train_data=train_data, test_data=test_data,
                hyperparams=hyperparams, hp_manager=hp_manager, objective_computer=objective.objective_computer())
            res.objective_on_test = computed_for_test.fitness()
            res.objective_on_train = computed_for_train.fitness()
            if compute_confidence:
                res.objective_on_test_ci = computed_for_test.ci95()
                res.objective_on_train_ci = computed_for_train.ci95()
    else:
        if objective.requires_predictions():
            computed_for_train = objective.objective_computer().compute_from_predictor_and_test_with_importance_mv(
                predictor=predictor, test_data=train_data.model_ready(),
                compute_confidence=compute_confidence)
            computed_for_test = objective.objective_computer().compute_from_predictor_and_test_with_importance_mv(
                predictor=predictor, test_data=test_data.model_ready(),
                compute_confidence=compute_confidence)
        else:  # Objective does not require predictions.
            computed_for_test, computed_for_train = validate_single_sample_and_structural_objective(
                train_data=train_data, test_data=test_data,
                hyperparams=hyperparams, hp_manager=hp_manager, objective_computer=objective.objective_computer())
        res.objective_on_train = computed_for_train.fitness()
        res.objective_on_test = computed_for_test.fitness()
        if compute_confidence:
            res.objective_on_train_ci = computed_for_train.ci95()
            res.objective_on_test_ci = computed_for_test.ci95()
    return res


def validate_single_fold_and_objective_sequential(
        train_data: ModelReadyInputData, test_data: ModelReadyInputData,
        predictors: Sequence[MVPredictor],
        hyperparams: Sequence, objective: SocialObjectiveWithImportance,
        compute_confidence: bool) -> AllFitnesses:
    """Passed x are multi-view. Passed predictors and hyperparams are one for each individual."""

    hp_manager = MaskMvHpManager.create_from_input_data(input_data=test_data)
    objective_on_test = []
    objective_on_train = []
    if compute_confidence:
        objective_on_test_ci = []
        objective_on_train_ci = []
    else:
        objective_on_test_ci = None
        objective_on_train_ci = None
    if objective.is_class_based():
        if objective.requires_predictions():
            y_train = train_data.the_outcome().data()
            y_test = test_data.the_outcome().data()
            for p, h in zip(predictors, hyperparams):
                pred_train = p.predict(train_data.views())
                pred_test = p.predict(test_data.views())
                computed_for_test = objective.compute_from_classes_with_confidence(
                    hyperparams=h, hp_manager=hp_manager, test_pred=pred_test,
                    test_true=y_test, train_pred=pred_train,
                    train_true=y_train, compute_confidence=compute_confidence)
                objective_on_test.append(computed_for_test.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(computed_for_test.ci95())
                if not objective.requires_training_predictions():
                    computed_for_train = objective.compute_from_classes_with_confidence(
                        hyperparams=h, hp_manager=hp_manager,
                        test_pred=pred_train, test_true=y_train,
                        train_pred=None, train_true=None, compute_confidence=compute_confidence)
                    objective_on_train.append(computed_for_train.fitness())
                    if compute_confidence:
                        objective_on_train_ci.append(computed_for_train.ci95())
        else:
            for h in hyperparams:
                computed_for_test, computed_for_train = validate_single_sample_and_structural_objective(
                    train_data=train_data, test_data=test_data,
                    hyperparams=h, hp_manager=hp_manager, objective_computer=objective.objective_computer())
                objective_on_test.append(computed_for_test.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(computed_for_test.ci95())
                objective_on_train.append(computed_for_train.fitness())
                if compute_confidence:
                    objective_on_train_ci.append(computed_for_train.ci95())
    else:
        if objective.requires_predictions():
            computed_for_train = objective.objective_computer().compute_from_predictor_and_test_with_importance_all_mv(
                predictors=predictors, test_data=train_data,
                compute_confidence=compute_confidence, seed=random_seed())
            computed_for_test = objective.objective_computer().compute_from_predictor_and_test_with_importance_all_mv(
                predictors=predictors, test_data=test_data,
                compute_confidence=compute_confidence, seed=random_seed())
            for c in computed_for_train:
                objective_on_train.append(c.fitness())
                if compute_confidence:
                    objective_on_train_ci.append(c.ci95())
            for c in computed_for_test:
                objective_on_test.append(c.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(c.ci95())
        else:  # Objective does not require predictions.
            for h in hyperparams:
                computed_for_test, computed_for_train = validate_single_sample_and_structural_objective(
                    train_data=train_data, test_data=test_data,
                    hyperparams=h, hp_manager=hp_manager, objective_computer=objective.objective_computer())
                objective_on_test.append(computed_for_test.fitness())
                objective_on_train.append(computed_for_train.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(computed_for_test.ci95())
                    objective_on_train_ci.append(computed_for_train.ci95())

    return AllFitnesses(
        test=objective_on_test, test_ci=objective_on_test_ci,
        train=objective_on_train, train_ci=objective_on_train_ci)


def validate_single_fold_and_objective(
        train_data: ModelReadyInputData,
        test_data: ModelReadyInputData,
        predictors: Sequence[MVPredictor],
        hyperparams: Sequence, objective: SocialObjectiveWithImportance,
        compute_confidence: bool, n_proc: int = 1) -> AllFitnesses:
    """Passed x are multi-view. Passed predictors and hyperparams are one for each individual."""
    n_predictors = len(predictors)
    cpu_count = multiprocessing.cpu_count()
    proc_to_use = max(1, min(n_proc, cpu_count, n_predictors))
    if proc_to_use == 1:
        return validate_single_fold_and_objective_sequential(
            train_data=train_data, test_data=test_data,
            predictors=predictors,
            hyperparams=hyperparams, objective=objective, compute_confidence=compute_confidence)
    else:
        worker_state = ValidateOnePredictorWorkerState(
            train_data=train_data.serialize(),
            test_data=test_data.serialize(),
            objective=objective,
            compute_confidence=compute_confidence)
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(
                processes=proc_to_use, initializer=predictor_validation_parallel_init,
                initargs=(worker_state,)) as workers_pool:
            results = [
                workers_pool.apply_async(predictor_multiprocessing_friendly_validation, args=(p,h))
                for p, h in zip(predictors, hyperparams)]
            results = [res.get() for res in results]  # Wait for all tasks to complete
        objective_on_test = [r.objective_on_test for r in results]
        objective_on_test_ci = [r.objective_on_test_ci for r in results]
        if len(objective_on_test_ci) == 0 or objective_on_test_ci[0] is None:
            objective_on_test_ci = None
        objective_on_train = [r.objective_on_train for r in results]
        objective_on_train_ci = [r.objective_on_train_ci for r in results]
        if len(objective_on_train_ci) == 0 or objective_on_train_ci[0] is None:
            objective_on_train_ci = None
        return AllFitnesses(
            test=objective_on_test, test_ci=objective_on_test_ci,
            train=objective_on_train, train_ci=objective_on_train_ci)


def evaluate_objective_for_fold_with_inner_cv(
        fold_predictors_with_hyperparams: MultiObjectiveOptimizerResult,
        objective: SocialObjectiveWithImportance,
        objective_index: int,
        train_data: InputData,
        test_data: InputData,
        compute_ci: bool = False, n_proc: int = 1) -> AllFitnesses:

    train_data = train_data.as_cached()
    test_data = test_data.as_cached()
    predictors = [p[objective_index] for p in fold_predictors_with_hyperparams.predictors()]
    hyperparams = fold_predictors_with_hyperparams.hyperparams()

    outcome_label = None
    if objective.has_outcome_label():
        outcome_label = objective.outcome_label()
    train_data = train_data.model_ready(outcome=outcome_label)
    test_data = test_data.model_ready(outcome=outcome_label)

    validate_single_fold_objective = validate_single_fold_and_objective(
        train_data=train_data, test_data=test_data,
        predictors=predictors, hyperparams=hyperparams, objective=objective,
        compute_confidence=compute_ci, n_proc=n_proc)

    if hyperparams[0].has_fitness():
        inner_cv_fold = get_fitnesses(pop=hyperparams, fitness_index=objective_index)
        inner_cv_fold_ci = get_ci95s(pop=hyperparams, fitness_index=objective_index)
        inner_cv_fold_sd = get_std_devs(pop=hyperparams, fitness_index=objective_index)
        inner_cv_fold_bootstrap_mean = get_bootstrap_means(pop=hyperparams, fitness_index=objective_index)
    else:
        inner_cv_fold = None
        inner_cv_fold_ci = None
        inner_cv_fold_sd = None
        inner_cv_fold_bootstrap_mean = None

    if validate_single_fold_objective.has_train_ci():
        train_ci = validate_single_fold_objective.train_ci()
    else:
        train_ci = None
    if validate_single_fold_objective.has_test_ci():
        test_ci = validate_single_fold_objective.test_ci()
    else:
        test_ci = None

    return AllFitnesses(
        test=validate_single_fold_objective.test(),
        test_ci=test_ci,
        train=validate_single_fold_objective.train(),
        train_ci=train_ci,
        inner_cv=inner_cv_fold,
        inner_cv_ci=inner_cv_fold_ci,
        inner_cv_sd=inner_cv_fold_sd,
        inner_cv_bootstrap_mean=inner_cv_fold_bootstrap_mean
        )


def evaluate_all_objectives_for_fold_with_inner_cv(
        fold_predictors_with_hyperparams: MultiObjectiveOptimizerResult,
        objectives: Sequence[SocialObjectiveWithImportance],
        train_data: InputData, test_data: InputData,
        compute_ci: bool = False, n_proc: int = 1) -> Sequence[AllFitnesses]:

    return [evaluate_objective_for_fold_with_inner_cv(
        fold_predictors_with_hyperparams=fold_predictors_with_hyperparams,
        objective=o,
        objective_index=i,
        train_data=train_data,
        test_data=test_data,
        compute_ci=compute_ci, n_proc=n_proc) for i, o in enumerate(objectives)]
