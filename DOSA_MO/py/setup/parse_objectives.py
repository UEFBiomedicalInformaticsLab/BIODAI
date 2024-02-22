from collections.abc import Sequence

from consts import DEFAULT_SURVIVAL_MODEL
from model.forest import ForestWithFallback, FOREST_NAME
from model.model import Model, DEFAULT_LOGISTIC_MAX_ITER, DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER
from model.model_with_coef import NBWithFallback
from model.logistic import LogisticWithFallback, DEFAULT_LOGISTIC_PENALTY
from model.survival_model import COX_NICK
from model.svm import SVMWithFallback
from model.tree import TreeWithFallback, TREE_NAME
from objective.objective_with_importance.objective_computer_with_importance import Accuracy, \
    BalancedAccuracy, MacroF1, ObjectiveComputerWithImportance
from objective.objective_with_importance.leanness import Leanness, SoftLeanness, RootLeanness
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance, \
    CompositePersonalObjectiveWithImportance
from objective.social_objective import PersonalObjective
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.plot_labels import SVM_LAB, LR_LAB, RF_LEGACY_LAB
from setup.allowed_names import NAIVE_BAYES_NAME, LOGISTIC_NAME, DEFAULT_MODEL_NAME
from util.utils import is_sequence_not_string


def parse_model(model_str: str, logistic_max_iter: int = DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER,
                penalty: str = DEFAULT_LOGISTIC_PENALTY) -> Model:
    if model_str == NAIVE_BAYES_NAME or model_str == NBWithFallback().nick():
        return NBWithFallback()
    elif model_str == LOGISTIC_NAME or model_str == LR_LAB:
        return LogisticWithFallback(max_iter=logistic_max_iter, penalty=penalty)
    elif model_str == TREE_NAME:
        return TreeWithFallback()
    elif model_str == FOREST_NAME or model_str == RF_LEGACY_LAB:
        return ForestWithFallback()
    elif model_str == SVMWithFallback().nick() or model_str == SVM_LAB:
        return SVMWithFallback()
    elif model_str == COX_NICK:
        return DEFAULT_SURVIVAL_MODEL
    else:
        raise ValueError("Unknown GA inner model: " + str(model_str))


def parse_objective_computer(
        objective_str: str) -> ObjectiveComputerWithImportance:
    """max_sd is used only if the selected objective is BalancedAccuracyWithDeviation"""
    if objective_str == Accuracy().nick():
        return Accuracy()
    elif objective_str == BalancedAccuracy().nick():
        return BalancedAccuracy()
    elif objective_str == MacroF1().nick():
        return MacroF1()
    elif objective_str == Leanness().nick():
        return Leanness()
    elif objective_str == SoftLeanness().nick():
        return SoftLeanness()
    elif objective_str == RootLeanness().nick():
        return RootLeanness()
    elif objective_str == CIndex().nick():
        return CIndex()
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_simple_objective(
        objective_str: str, target: str, use_model: bool,
        model: Model = None) -> PersonalObjectiveWithImportance:
    """Model and target string are provided by the caller.
    use_model is true if the main algorithm in general uses inner models, but it is still possible that
    the objective does not use an inner model."""
    objective_computer = parse_objective_computer(objective_str=objective_str)
    if use_model and objective_computer.requires_predictions():
        if model is None:
            if objective_computer.is_class_objective_computer():
                model = parse_model(DEFAULT_MODEL_NAME)  # Use default
            elif objective_computer.is_survival_objective_computer():
                model = DEFAULT_SURVIVAL_MODEL
            else:
                raise Exception("Unexpected type of objective computer: " + str(objective_computer))
    else:
        model = None
    return CompositePersonalObjectiveWithImportance(
        objective_computer=objective_computer, target_label=target, model=model)


def parse_composite_objective(objective_str: Sequence, default_target: str, use_model: bool,
                              logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                              penalty: str = DEFAULT_LOGISTIC_PENALTY
                              ) -> PersonalObjective:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that
    the objective does not use an inner model."""
    len_s = len(objective_str)
    if len_s > 0:
        objective_class_nick = objective_str[0]
        model = None
        target_label = default_target
        if len_s > 1:
            model = parse_model(objective_str[1], logistic_max_iter=logistic_max_iter, penalty=penalty)
            if len_s > 2:
                target_label = objective_str[2]
    else:
        raise ValueError("Empty objective.")
    return parse_simple_objective(
        objective_str=objective_class_nick, model=model, target=target_label, use_model=use_model)


def parse_objective(objective_str, default_target: str, use_model: bool,
                    logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                    penalty: str = DEFAULT_LOGISTIC_PENALTY) -> PersonalObjective:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that
        the objective does not use an inner model."""
    if is_sequence_not_string(objective_str):
        return parse_composite_objective(
            objective_str=objective_str, default_target=default_target, use_model=use_model,
            logistic_max_iter=logistic_max_iter, penalty=penalty)
    elif isinstance(objective_str, str):
        return parse_simple_objective(
            objective_str=objective_str, target=default_target, use_model=use_model)
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_objectives(objectives_str: Sequence[str], default_target: str, use_model: bool,
                     logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                     penalty: str = DEFAULT_LOGISTIC_PENALTY) -> list[PersonalObjective]:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that some
    objectives do not use an inner model."""
    objectives = []
    for s in objectives_str:
        objectives.append(
            parse_objective(
                s, default_target=default_target, use_model=use_model,
                logistic_max_iter=logistic_max_iter,
                penalty=penalty))
    return objectives
