from collections.abc import Sequence

from model.forest import ForestWithFallback
from model.model import Model, DEFAULT_LOGISTIC_MAX_ITER
from model.model_with_coef import NBWithFallback, LogisticWithFallback
from objective.composite_personal_objective import CompositePersonalObjective
from objective.objective_computer import Accuracy, BalancedAccuracy, Leanness, ObjectiveComputer
from objective.social_objective import PersonalObjective
from setup.allowed_names import NAIVE_BAYES_NAME, LOGISTIC_NAME, DEFAULT_MODEL_NAME, FOREST_NAME
from util.utils import is_sequence_not_string


def parse_model(model_str: str, logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER) -> Model:
    if model_str == NAIVE_BAYES_NAME:
        return NBWithFallback()
    elif model_str == LOGISTIC_NAME:
        return LogisticWithFallback(max_iter=logistic_max_iter)
    elif model_str == FOREST_NAME:
        return ForestWithFallback()
    else:
        raise ValueError("Unknown GA inner model.")


def parse_objective_computer(
        objective_str: str) -> ObjectiveComputer:
    if objective_str == Accuracy().nick():
        return Accuracy()
    elif objective_str == BalancedAccuracy().nick():
        return BalancedAccuracy()
    elif objective_str == Leanness().nick():
        return Leanness()
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_simple_objective(
        objective_str: str, target: str, use_model: bool,
        model: Model = None) -> PersonalObjective:
    objective_computer = parse_objective_computer(objective_str=objective_str)
    if use_model:
        if model is None:
            if objective_computer.is_class_objective_computer():
                model = parse_model(DEFAULT_MODEL_NAME)  # Use default
            else:
                raise Exception("Unexpected type of objective computer.")
        return CompositePersonalObjective(objective_computer=objective_computer, target_label=target, model=model)
    else:
        if objective_computer.requires_predictions():
            return CompositePersonalObjective(objective_computer=objective_computer, target_label=target)
        else:
            return CompositePersonalObjective(objective_computer=objective_computer)


def parse_composite_objective(objective_str: Sequence, default_target: str, use_model: bool,
                              logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER
                              ) -> PersonalObjective:
    len_s = len(objective_str)
    if len_s > 0:
        objective_class_nick = objective_str[0]
        model = None
        target_label = default_target
        if len_s > 1:
            model = parse_model(objective_str[1], logistic_max_iter=logistic_max_iter)
            if len_s > 2:
                target_label = objective_str[2]
    else:
        raise ValueError("Empty objective.")
    return parse_simple_objective(
        objective_str=objective_class_nick, model=model, target=target_label, use_model=use_model)


def parse_objective(objective_str, default_target: str, use_model: bool,
                    logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER) -> PersonalObjective:
    if is_sequence_not_string(objective_str):
        return parse_composite_objective(
            objective_str=objective_str, default_target=default_target, use_model=use_model,
            logistic_max_iter=logistic_max_iter)
    elif isinstance(objective_str, str):
        return parse_simple_objective(
            objective_str=objective_str, target=default_target, use_model=use_model)
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_objectives(objectives_str: Sequence, default_target: str, use_model: bool,
                     logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER) -> [PersonalObjective]:
    objectives = []
    for s in objectives_str:
        objectives.append(
            parse_objective(
                s, default_target=default_target, use_model=use_model,
                logistic_max_iter=logistic_max_iter))
    return objectives
