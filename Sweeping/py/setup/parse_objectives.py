from collections.abc import Sequence
from typing import Optional

from consts import DEFAULT_SURVIVAL_MODEL
from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from model.classification.forest import FOREST_NAME, ForestFactory

from model.class_crisp.classifier_with_coef import SklearnCrispClassModelCreator
from model.classification.naive_bayes import NB_NICK, NBFactory
from model.class_proba.sv_proba import SklearnProbaClassModelCreator
from model.classification.svm import RBF_SVM_NICK, RSVMFactory, LINEAR_SVM_NICK, LSVMFactory
from model.classification.xgboost import XGBOOST_NICK, XGBoostFactory
from model.multi_view.adjusted_mv_model import AdjustedMVModel
from model.multi_view.multi_view_model import MVModel, SvToMvModelWrapper
from model.sv_model import SVModel
from model.classification.logistic import DEFAULT_LOGISTIC_MAX_ITER, DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER, \
    DEFAULT_LOGISTIC_PENALTY, LogisticFactory
from model.survival.sksurv_model import SksurvModel
from model.survival.survival_model import COX_NICK, LifelinesModel
from model.classification.tree import TREE_NAME, TreeFactory
from objective.balanced_accuracy_with_deviation import BalancedAccuracyWithDeviation, DEFAULT_MAX_DEVIATION
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.separation.min_separation import MinSeparation
from objective.objective_with_importance.objective_computer_with_importance import Accuracy, \
    BalancedAccuracy, MacroF1, ObjectiveComputerWithImportance, BRIER_NICK, BrierScore, AUC_NICK, AUC, AUCPR_NICK, AUCPR
from objective.objective_with_importance.leanness import Leanness, SoftLeanness, RootLeanness
from objective.objective_with_importance.separation.root_separation import RootSeparation
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance, \
    CompositePersonalObjectiveWithImportance
from objective.social_objective import PersonalObjective
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.plot_labels import SVM_LAB, LR_LAB, RF_LEGACY_LAB
from setup.allowed_names import NAIVE_BAYES_NAME, LOGISTIC_NAME, DEFAULT_MODEL_NAME
from util.utils import is_sequence_not_string
from view_adjuster.view_adjuster_model import DEFAULT_VIEW_ADJUSTER_MODEL
from view_adjuster.views_adjuster import UniformViewsAdjusterModel


def parse_model(model_str: str,
                objective_computer: ObjectiveComputer,
                logistic_max_iter: int = DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER,
                penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY) -> SVModel:
    model_creator = None
    if objective_computer.is_crisp_class_objective_computer():
        model_creator = SklearnCrispClassModelCreator()
    elif objective_computer.is_proba_class_objective_computer():
        model_creator = SklearnProbaClassModelCreator()
    else:
        ValueError()
    if model_str == NAIVE_BAYES_NAME or model_str == NB_NICK:
        return model_creator.create_model(model_factory=NBFactory())
    elif model_str == LOGISTIC_NAME or model_str == LR_LAB:
        return model_creator.create_model(model_factory=LogisticFactory(max_iter=logistic_max_iter, penalty=penalty))
    elif model_str == TREE_NAME:
        return model_creator.create_model(model_factory=TreeFactory())
    elif model_str == FOREST_NAME:
        return model_creator.create_model(model_factory=ForestFactory())
    elif model_str == RF_LEGACY_LAB:
        return model_creator.create_model(model_factory=ForestFactory(min_samples_leaf=2))
    elif model_str == RBF_SVM_NICK or model_str == SVM_LAB:
        return model_creator.create_model(
            model_factory=RSVMFactory(probability=objective_computer.is_proba_class_objective_computer()))
    elif model_str == LINEAR_SVM_NICK:
        return model_creator.create_model(
            model_factory=LSVMFactory(probability=objective_computer.is_proba_class_objective_computer()))
    elif model_str == XGBOOST_NICK:
        return model_creator.create_model(model_factory=XGBoostFactory())
    elif model_str == COX_NICK:
        return DEFAULT_SURVIVAL_MODEL
    elif model_str == SksurvModel().nick():
        return SksurvModel()
    elif model_str == LifelinesModel().nick():
        return LifelinesModel()
    else:
        raise ValueError("Unknown inner model: " + str(model_str))


def parse_model_mv(model_str: str,
                objective_computer: ObjectiveComputer,
                logistic_max_iter: int = DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER,
                penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY) -> MVModel:
    return AdjustedMVModel(views_adjuster_model=UniformViewsAdjusterModel(adjuster_model=DEFAULT_VIEW_ADJUSTER_MODEL),
                           inner_model=SvToMvModelWrapper(
                               sv_model=parse_model(
                                   model_str=model_str, objective_computer=objective_computer,
                                   logistic_max_iter=logistic_max_iter, penalty=penalty)))


def parse_objective_computer(
        objective_str: str,
        max_sd: float = DEFAULT_MAX_DEVIATION,
        classes: Optional[Sequence[str]] = None) -> ObjectiveComputerWithImportance:
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
    elif objective_str == MinSeparation().nick():
        return MinSeparation()
    elif objective_str == RootSeparation().nick():
        return RootSeparation()
    elif objective_str == BRIER_NICK:
        return BrierScore()
    elif objective_str == AUC_NICK:
        return AUC()
    elif objective_str == AUCPR_NICK:
        return AUCPR()
    # elif objective_str == NotIBS().nick():
    #     return NotIBS()
    # IBS is problematic because we need also to set the time interval considered.
    elif objective_str == BalancedAccuracyWithDeviation().base_nick():
        raise NotImplementedError("Not supported at the moment.")
        # return BalancedAccuracyWithDeviation(max_sd=max_sd)
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_simple_objective(
        objective_str: str, target: str, use_model: bool,
        outcomes: Sequence[Outcome],
        max_sd: float = DEFAULT_MAX_DEVIATION,
        model: MVModel = None) -> PersonalObjectiveWithImportance:
    """Model and target string are provided by the caller.
    use_model is true if the main algorithm in general uses inner models, but it is still possible that
    the objective does not use an inner model."""
    objective_computer = parse_objective_computer_from_outcomes(
        objective_str=objective_str, max_sd=max_sd, outcomes=outcomes, target_label=target)
    if use_model and objective_computer.requires_predictions():
        if model is None:
            if objective_computer.is_class_objective_computer():
                model = parse_model_mv(model_str=DEFAULT_MODEL_NAME, objective_computer=objective_computer)  # Use default
            elif objective_computer.is_survival_objective_computer():
                model = DEFAULT_SURVIVAL_MODEL
            else:
                raise Exception("Unexpected type of objective computer: " + str(objective_computer))
    else:
        model = None
    return CompositePersonalObjectiveWithImportance(
        objective_computer=objective_computer, target_label=target, model=model)


def parse_objective_computer_from_outcomes(
        objective_str: str, max_sd: float,
        outcomes: Sequence[Outcome], target_label: str) -> ObjectiveComputerWithImportance:
    outcome = None
    for o in outcomes:
        if o.name() == target_label:
            outcome = o
    classes = None
    if outcome is not None and outcome.type() == OutcomeType.categorical:
        classes = outcome.class_labels()
    return parse_objective_computer(objective_str=objective_str, max_sd=max_sd, classes=classes)



def parse_composite_objective(objective_str: Sequence, default_target: str, use_model: bool, max_sd: float,
                              outcomes: Sequence[Outcome],
                              logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                              penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY
                              ) -> PersonalObjective:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that
    the objective does not use an inner model."""
    len_s = len(objective_str)
    if len_s > 0:
        objective_class_nick = objective_str[0]
        model = None
        target_label = default_target
        objective_computer = parse_objective_computer_from_outcomes(
            objective_str=objective_class_nick, max_sd=max_sd, outcomes=outcomes, target_label=target_label)
        if len_s > 1:
            model = parse_model_mv(model_str=objective_str[1], objective_computer=objective_computer,
                                logistic_max_iter=logistic_max_iter, penalty=penalty)
            if len_s > 2:
                target_label = objective_str[2]
    else:
        raise ValueError("Empty objective.")
    return parse_simple_objective(
        objective_str=objective_class_nick, model=model, target=target_label, use_model=use_model, max_sd=max_sd,
        outcomes=outcomes)


def parse_objective(objective_str, default_target: str, use_model: bool, max_sd: float,
                    outcomes: Sequence[Outcome],
                    logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                    penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY) -> PersonalObjective:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that
        the objective does not use an inner model."""
    if is_sequence_not_string(objective_str):
        return parse_composite_objective(
            objective_str=objective_str, default_target=default_target, use_model=use_model, max_sd=max_sd,
            logistic_max_iter=logistic_max_iter, penalty=penalty, outcomes=outcomes)
    elif isinstance(objective_str, str):
        return parse_simple_objective(
            objective_str=objective_str, target=default_target, use_model=use_model, max_sd=max_sd, outcomes=outcomes)
    else:
        raise ValueError("Unknown objective: " + str(objective_str))


def parse_objectives(objectives_str: Sequence[str], default_target: str, use_model: bool, max_sd: float,
                     outcomes: Sequence[Outcome],
                     logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                     penalty: Optional[str] = DEFAULT_LOGISTIC_PENALTY
                     ) -> list[PersonalObjectiveWithImportance]:
    """use_model is true if the main algorithm in general uses inner models, but it is still possible that some
    objectives do not use an inner model."""
    objectives = []
    for s in objectives_str:
        objectives.append(
            parse_objective(
                s, default_target=default_target, use_model=use_model, max_sd=max_sd,
                logistic_max_iter=logistic_max_iter,
                penalty=penalty, outcomes=outcomes))
    return objectives
