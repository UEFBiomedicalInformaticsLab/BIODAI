from typing import NamedTuple
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from model.model import ClassModel, Classifier
from objective.social_objective import SocialObjective


def select_by_indices(samples, indices):
    return samples.iloc[indices]


# Each fold is a pair training-testing
def select_all_sets(x, y, fold):
    train_indices = fold[0]
    test_indices = fold[1]
    x_train = select_by_indices(x, train_indices)
    y_train = select_by_indices(y, train_indices)
    x_test = select_by_indices(x, test_indices)
    y_test = select_by_indices(y, test_indices)
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


class ValidateSingleFoldAndObjectiveRes(NamedTuple):
    """Returns lists with a value for each individual."""
    objective_on_test: list[float]
    objective_on_train: list[float]


def validate_single_fold_and_objective(
        x_train, y_train, x_test, y_test, predictors: [Classifier], hyperparams, objective: SocialObjective):
    """Passed x are multi-view. Passed predictors and hyperparams are one for each individual."""

    hp_manager = DummyHpManager()
    objective_on_test = []
    objective_on_train = []
    if objective.is_class_based():
        if objective.requires_predictions():
            for p, h in zip(predictors, hyperparams):
                pred_train = p.predict(x_train)
                pred_test = p.predict(x_test)
                objective_on_test.append(objective.compute_from_classes(
                    hyperparams=h, hp_manager=hp_manager, test_pred=pred_test,
                    test_true=y_test, train_pred=pred_train,
                    train_true=y_train))
                if not objective.requires_training_predictions():
                    objective_on_train.append(objective.compute_from_classes(hyperparams=h, hp_manager=hp_manager,
                                                                             test_pred=pred_train, test_true=y_train,
                                                                             train_pred=None, train_true=None))
        else:
            for h in hyperparams:
                computed_objective = objective.compute_from_classes(
                    hyperparams=h, hp_manager=hp_manager, test_pred=None, test_true=None, train_pred=None,
                    train_true=None)
                objective_on_test.append(computed_objective)
                objective_on_train.append(computed_objective)
    else:
        for p, h in zip(predictors, hyperparams):
            objective_on_train.append(
                objective.objective_computer().compute_from_predictor_and_test(
                    predictor=p, x_test=x_train, y_test=y_train))
            objective_on_test.append(
                objective.objective_computer().compute_from_predictor_and_test(
                    predictor=p, x_test=x_test, y_test=y_test))

    return ValidateSingleFoldAndObjectiveRes(
        objective_on_test=objective_on_test, objective_on_train=objective_on_train)
