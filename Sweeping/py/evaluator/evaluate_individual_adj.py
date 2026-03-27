from collections.abc import Sequence

from cross_validation.multi_objective.mo_cv_result import MOCVResult
from cross_validation.single_objective.cv_result import CVResult
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from input_data.evaluation_ready_input_data import NoOutcomesInputData
from input_data.input_data import InputData
from model.multi_view.multi_view_model import MVModel
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.randoms import set_all_seeds
from util.utils import IllegalStateError


def evaluate_objective_adj(
        selected_input_data: InputData,
        folds_list: list[tuple[IndexArray, IndexArray]],
        hp_manager: MvHyperparamManager,
        individual: PeculiarIndividualByListlike,
        objective: PersonalObjectiveWithImportance,
        seed: int,
        compute_feature_importance: bool,
        compute_confidence: bool) -> CVResult:
    """Returns the fitnesses for the defined objectives, and the related predictors.
    The predictors are fitted on all the samples passed.
    Passed input data must be already selected with only the used features.
    The returned predictor are able to select the features they need from the views considering the individual."""
    set_all_seeds(seed)
    cv_result = None
    objective_computer = objective.objective_computer()
    if objective_computer.requires_target():
        selected_input_data = selected_input_data.model_ready(outcome=objective.outcome_label())
    else:
        selected_input_data = NoOutcomesInputData(
            all_views=selected_input_data.views(),
            adjusted_views=selected_input_data.adjusted_view_def(),
            nick=selected_input_data.nick())
    if objective.requires_predictions():  # If requires predictions it also requires targets.
        if objective.has_model():
            model = objective.mv_model()
            if not isinstance(model, MVModel):
                raise ValueError("The provided model is not a MVModel.\n" +
                                 "Provided model:\n" +
                                 str(model) + "\n")
            assert isinstance(model, MVModel)
            cv_result = objective_computer.compute_with_kfold_cv_with_importance_mv(
                model=model, data=selected_input_data, folds_list=folds_list,
                compute_fi=compute_feature_importance,
                compute_confidence=compute_confidence)
            inner_predictor = model.checked_fit(data=selected_input_data)
            lifter = hp_manager.feature_space_lifter(hyperparams=individual)
            downlifted_predictor = inner_predictor.downlift(lifter=lifter)
            cv_result.set_final_predictor(predictor=downlifted_predictor)
        else:
            ValueError("Unexpected case.")
    else:
        cv_result = objective_computer.compute_from_structure_with_importance(
            hyperparams=individual,
            hp_manager=hp_manager,
            data=selected_input_data,
            compute_fi=compute_feature_importance,
            compute_confidence=compute_confidence
        )
    if cv_result is None:
        raise IllegalStateError("A None result was produced.")
    return cv_result


def evaluate_individual_adj(
        input_data: InputData,
        folds_list: list[tuple[IndexArray,IndexArray]],
        hp_manager: MvHyperparamManager,
        individual: PeculiarIndividualByListlike,
        objectives: Sequence[PersonalObjectiveWithImportance],
        seed: int,
        compute_feature_importance: bool,
        compute_confidence: bool) -> MOCVResult:
    """Returns the fitnesses for the defined objectives, and the related predictors.
            The predictors are fitted on all the samples passed.
            Returned predictors are able to select the features they need from the views considering the individual
            they are created from."""
    selected_input_data = input_data.select_features(masks=hp_manager.used_feature_masks(hyperparams=individual))
    so_results = [evaluate_objective_adj(
        selected_input_data=selected_input_data, folds_list=folds_list, hp_manager=hp_manager,
        individual=individual, objective=o, seed=seed,
        compute_feature_importance=compute_feature_importance,
        compute_confidence=compute_confidence) for o in objectives]
    return MOCVResult(so_results=so_results)
