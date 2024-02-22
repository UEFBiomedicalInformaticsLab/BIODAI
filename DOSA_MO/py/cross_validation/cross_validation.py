from collections.abc import Sequence

from pandas import DataFrame

from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.single_objective.all_fold_fitnesses import AllFitnesses
from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.confident_individual import get_ci95s, get_std_devs, get_bootstrap_means
from individual.fit_individual import get_fitnesses
from model.model import ClassModel, Classifier
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from objective.objective_with_importance.social_objective_with_importance import SocialObjectiveWithImportance
from prediction_stats.stat_creator import StatCreator
from util.dataframes import select_by_row_indices
from views.views import Views


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
def cross_validate(x, y, folds_list, model: ClassModel):
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
def cross_validate_and_create_stats(x, y, folds_list, model: ClassModel, stat_creator: StatCreator):
    train_pred_y, train_true_y, test_pred_y, test_true_y = cross_validate(x=x, y=y, folds_list=folds_list, model=model)
    return stat_creator.create_stats(
        test_predicted_y=test_pred_y, test_true_y=test_true_y,
        train_predicted_y=train_pred_y, train_true_y=train_true_y)


def validate_single_sample_and_structural_objective(
        x_train: Views, y_train, x_test: Views, y_test,
        hyperparams, hp_manager, objective_computer: ObjectiveComputerWithImportance) -> tuple[CVResult, CVResult]:
    computed_for_test = objective_computer.compute_from_structure_with_importance(
        hyperparams=hyperparams, hp_manager=hp_manager,
        x=x_test.collapsed_filtered_by_mask(mask=hp_manager.active_features_mask(hyperparams=hyperparams)),
        y=y_test, compute_confidence=True, compute_fi=False)
    computed_for_train = objective_computer.compute_from_structure_with_importance(
        hyperparams=hyperparams, hp_manager=hp_manager,
        x=x_train.collapsed_filtered_by_mask(mask=hp_manager.active_features_mask(hyperparams=hyperparams)),
        y=y_train,
        compute_confidence=True, compute_fi=False)
    return computed_for_test, computed_for_train


def validate_single_fold_and_objective(
        x_train: Views, y_train, x_test: Views, y_test, predictors: [Classifier],
        hyperparams, objective: SocialObjectiveWithImportance,
        compute_confidence: bool) -> AllFitnesses:
    """Passed x are multi-view. Passed predictors and hyperparams are one for each individual."""

    hp_manager = DummyHpManager()
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
            for p, h in zip(predictors, hyperparams):
                pred_train = p.predict(x_train)
                pred_test = p.predict(x_test)
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
                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                    hyperparams=h, hp_manager=hp_manager, objective_computer=objective.objective_computer())
                objective_on_test.append(computed_for_test.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(computed_for_test.ci95())
                objective_on_train.append(computed_for_train.fitness())
                if compute_confidence:
                    objective_on_train_ci.append(computed_for_train.ci95())
    else:
        if objective.requires_predictions():
            computed_for_train = objective.objective_computer().compute_from_predictor_and_test_with_importance_all(
                predictors=predictors, x_test=x_train, y_test=y_train,
                compute_fi=False, compute_confidence=compute_confidence)
            computed_for_test = objective.objective_computer().compute_from_predictor_and_test_with_importance_all(
                predictors=predictors, x_test=x_test, y_test=y_test,
                compute_fi=False, compute_confidence=compute_confidence)
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
                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                    hyperparams=h, hp_manager=hp_manager, objective_computer=objective.objective_computer())
                objective_on_test.append(computed_for_test.fitness())
                if compute_confidence:
                    objective_on_test_ci.append(computed_for_test.ci95())
                objective_on_train.append(computed_for_train.fitness())
                if compute_confidence:
                    objective_on_train_ci.append(computed_for_train.ci95())

    return AllFitnesses(
        test=objective_on_test, test_ci=objective_on_test_ci,
        train=objective_on_train, train_ci=objective_on_train_ci)


def evaluate_objective_for_fold_with_inner_cv(
        fold_predictors_with_hyperparams: MultiObjectiveOptimizerResult,
        objective: SocialObjectiveWithImportance,
        objective_index: int,
        x_train: Views, y_train: dict[str, DataFrame], x_test: Views, y_test: dict[str, DataFrame],
        compute_ci: bool = False) -> AllFitnesses:

    x_train = x_train.as_cached()
    x_test = x_test.as_cached()
    predictors = [p[objective_index] for p in fold_predictors_with_hyperparams.predictors()]
    hyperparams = fold_predictors_with_hyperparams.hyperparams()

    outcome_label = None
    if objective.has_outcome_label():
        outcome_label = objective.outcome_label()

    obj_y_train = None
    obj_y_test = None
    if outcome_label is not None:
        obj_y_train = y_train[outcome_label]
        obj_y_test = y_test[outcome_label]

    validate_single_fold_objective = validate_single_fold_and_objective(
        x_train, obj_y_train, x_test, obj_y_test,
        predictors, hyperparams, objective, compute_confidence=compute_ci)

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
        x_train: Views, y_train: dict[str, DataFrame], x_test: Views, y_test: dict[str, DataFrame],
        compute_ci: bool = False) -> Sequence[AllFitnesses]:

    return [evaluate_objective_for_fold_with_inner_cv(
        fold_predictors_with_hyperparams=fold_predictors_with_hyperparams,
        objective=o,
        objective_index=i,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        compute_ci=compute_ci) for i, o in enumerate(objectives)]
