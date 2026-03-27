from collections.abc import Sequence

from pandas import DataFrame

from cross_validation.multi_objective.mo_cv_result import MOCVResult
from evaluator.evaluate_individual_adj import evaluate_objective_adj
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.sv_to_mv_hp_manager_wrapper import SvToMvHpManagerWrapper
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from input_data.input_data import InputData
from input_data.model_ready_input_data import ModelReadyInputData
from input_data.outcome import smart_create_outcome
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.dataframe.dataframes import has_non_finite_error
from util.table.table import Table
from views.views import JustViews


def collapsed_views_filtered(
        collapsed_views: Table,
        hp_manager: SvHyperparamManager,
        individual: PeculiarIndividualByListlike) -> Table:
    mask = hp_manager.collapsed_used_features_mask(hyperparams=individual)
    return collapsed_views.filter_cols_by_mask(mask=mask)


def fit_inner_model(
        train_filtered_data: ModelReadyInputData, model: MVModel, check_training_data: bool = False) -> MVPredictor:
    """Columns are assumed to be already filtered. Differently from calling model.fit directly, this
    method checks the input if check_training_data is True."""
    if check_training_data:  # Debug check.
        if train_filtered_data is None:
            raise ValueError("train_filtered_data is None")
        if train_filtered_data.has_non_finite_x():
            raise has_non_finite_error(df=train_filtered_data.views().to_dataframe())
    fit_model = model.fit(data=train_filtered_data)
    return fit_model


def evaluate_individual(
        collapsed_views: Table,
        outcomes: dict[str, DataFrame],
        folds_list: list[tuple[IndexArray,IndexArray]],
        hp_manager: SvHyperparamManager,
        individual: PeculiarIndividualByListlike,
        objectives: Sequence[PersonalObjectiveWithImportance],
        seed: int,
        compute_feature_importance: bool,
        compute_confidence: bool) -> MOCVResult:
    """Returns the fitnesses for the defined objectives, and the related predictors.
            The predictors are fitted on all the samples passed.
            """
    outcomes = [smart_create_outcome(y=o, name=n) for n,o in outcomes.items()]
    input_data = InputData.smart_create(
        all_views = JustViews(views_dict={"x": collapsed_views}), outcomes=outcomes, nick="data")
    mv_hp_manager = SvToMvHpManagerWrapper(
        sv_hp_manager=hp_manager,
        view_name=input_data.view_names_seq()[0],
        predictive_features_num=hp_manager.predictive_features_mask_len(hyperparams=individual)
    )
    selected_input_data = input_data.select_features(masks=mv_hp_manager.used_feature_masks(hyperparams=individual))
    so_results = [evaluate_objective_adj(
        selected_input_data=selected_input_data, folds_list=folds_list, hp_manager=mv_hp_manager,
        individual=individual, objective=o, seed=seed,
        compute_feature_importance=compute_feature_importance,
        compute_confidence=compute_confidence) for o in objectives]
    return MOCVResult(so_results=so_results)
