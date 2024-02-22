from cross_validation.multi_objective.optimizer.mo_optimizer_factory import create_mo_optimizer_by_fold
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold
from ga_components.bitlist_mutation import FlipMutation, SymmetricFlipMutation
from ga_components.selection import ElitistSelection, TournamentExtraction
from ga_components.sorter.sorting_strategy import SortingStrategyCrowd, \
    SortingStrategyCrowdFull, SortingStrategyCrowdCI, \
    SortingStrategyNsga3, SortingStrategyNsga3CI
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from objective.social_objective import PersonalObjective
from setup.allowed_names import CROWDING_DISTANCE_NAME, \
    CROWDING_DISTANCE_FULL_NAME, CROWDING_DISTANCE_CI_NAME, LASSO_NAME, NSGA_STAR_NAME, \
    LASSO_MO_NAME, NSGA3_NAME, PAM50_NAME, NSGA3_CI_NAME, ADJUSTED_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import big_nsga_setup, \
    small_nsga_setup, big_adjusted_setup, small_adjusted_setup
from setup.parse_feature_importance import parse_feature_importance_by_fold
from setup.parse_initial_features import parse_initial_features
from setup.parse_objectives import parse_objectives
from setup.setup_utils import load_input_data
from univariate_feature_selection.feature_selector import AnovaAndCoxFeatureSelector
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMOUnion
from util.printer.printer import Printer


def setup_to_mo_optimizer(setup: EvaluationSetup, printer: Printer
                          ) -> tuple[MultiObjectiveOptimizerByFold, InputData, list[PersonalObjective]]:

    dataset_name = setup.dataset()
    mvmo_algorithm = setup.mvmo_algorithm()
    use_big_setup = setup.use_big_defaults()
    mating_prob = setup.mating_prob()
    mutation_frequency = setup.mutation_frequency()
    sorting_strategy_str = setup.sorting_strategy()
    generations = setup.generations()
    views_to_use = setup.views_to_use()
    pop = setup.pop()
    inner_n_folds = setup.inner_n_folds()

    selection_str = setup.selection()
    if selection_str == ElitistSelection().name():
        selection = ElitistSelection()
    elif selection_str == TournamentExtraction.base_nick():
        selection = TournamentExtraction(n_participants=setup.selection_tournament_size())
    else:
        raise ValueError("Unknown selection: " + selection_str)

    input_data = load_input_data(dataset_name=dataset_name, views_to_use=views_to_use, printer=printer)

    run_nsga = (mvmo_algorithm == NSGA_STAR_NAME)
    run_sweeping_ga = False
    run_adjusted = mvmo_algorithm == ADJUSTED_NAME

    if mvmo_algorithm == LASSO_NAME or mvmo_algorithm == LASSO_MO_NAME:
        use_inner_model = False
    elif (run_nsga or
          run_sweeping_ga or
          mvmo_algorithm == PAM50_NAME or
          run_adjusted):
        use_inner_model = True
    else:
        raise ValueError("Unknown MVMO algorithm.")

    objectives = parse_objectives(
        objectives_str=setup.objectives(),
        default_target=input_data.stratify_outcome_name(),
        use_model=use_inner_model,
        logistic_max_iter=setup.logistic_max_iter(),
        penalty=setup.penalty())
    printer.print_variable("Objectives", objectives)

    n_objectives = len(objectives)

    if sorting_strategy_str == CROWDING_DISTANCE_NAME:
        sorting_strategy = SortingStrategyCrowd(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_FULL_NAME:
        sorting_strategy = SortingStrategyCrowdFull(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_CI_NAME:
        sorting_strategy = SortingStrategyCrowdCI(selection=selection)
    elif sorting_strategy_str == NSGA3_NAME:
        sorting_strategy = SortingStrategyNsga3(
            selection=selection, num_objectives=n_objectives, max_reference_points=pop)
    elif sorting_strategy_str == NSGA3_CI_NAME:
        sorting_strategy = SortingStrategyNsga3CI(
            selection=selection, num_objectives=n_objectives, max_reference_points=pop)
    else:
        raise ValueError("Unknown sorting strategy: " + sorting_strategy_str)

    feature_importance_by_fold = parse_feature_importance_by_fold(
        categorical_fi_str=setup.feature_importance_categorical(),
        survival_fi_str=setup.feature_importance_survival(),
        base_dir=setup.load_base_dir(),
        printer=printer)

    printer.print("Removing outcomes not necessary for objectives from input data.")
    input_data = select_outcomes_in_objectives(input_data=input_data, objectives=objectives)

    bitlist_mutation_str = setup.bitlist_mutation_operator()
    if bitlist_mutation_str == FlipMutation().nick():
        bitlist_mutation = FlipMutation()
    elif bitlist_mutation_str == SymmetricFlipMutation().nick():
        bitlist_mutation = SymmetricFlipMutation()
    else:
        raise ValueError("Unknown bitlist mutation operator.")

    initial_features = parse_initial_features(
        initial_features_strategy_str=setup.initial_features_strategy(),
        initial_features_min=setup.initial_features_min(),
        initial_features_max=setup.initial_features_max())

    outcome_keys = set()
    for o in objectives:
        if o.requires_predictions():
            outcome_keys.add(o.outcome_label())

    # Using GAs
    feature_selector_mo = FeatureSelectorMOUnion(feature_selector_so=AnovaAndCoxFeatureSelector())
    if use_big_setup:
        if run_nsga:
            mo_optimizer = big_nsga_setup(
                objectives=objectives,
                pop_size=pop,
                initial_features=initial_features,
                n_gen=generations.concatenated_generations(),
                mating_prob=mating_prob, mutating_prob=mutation_frequency,
                sorting_strategy=sorting_strategy,
                inner_n_folds=inner_n_folds,
                mutation=bitlist_mutation,
                use_clone_repurposing=setup.use_clone_repurposing())
        elif run_adjusted:
            mo_optimizer = big_adjusted_setup(
                objectives=objectives,
                pop_size=pop,
                initial_features=initial_features,
                n_gen=generations.concatenated_generations(),
                mating_prob=mating_prob, mutation_frequency=mutation_frequency,
                sorting_strategy=sorting_strategy,
                outer_n_folds=setup.outer_n_folds(),
                inner_n_folds=inner_n_folds,
                mutation=bitlist_mutation,
                use_clone_repurposing=setup.use_clone_repurposing(),
                adjuster_regressor=setup.adjuster_regressor())
        else:
            raise ValueError("Unknown MVMO algorithm.")
    else:
        if run_nsga:
            mo_optimizer = small_nsga_setup(
                objectives=objectives,
                pop_size=pop,
                initial_features=initial_features,
                n_gen=generations.concatenated_generations(),
                mating_prob=mating_prob, mutating_prob=mutation_frequency,
                sorting_strategy=sorting_strategy,
                inner_n_folds=inner_n_folds,
                mutation=bitlist_mutation,
                use_clone_repurposing=setup.use_clone_repurposing())
        elif run_adjusted:
            mo_optimizer = small_adjusted_setup(
                objectives=objectives,
                pop_size=pop,
                initial_features=initial_features,
                n_gen=generations.concatenated_generations(),
                mating_prob=mating_prob, mutation_frequency=mutation_frequency,
                sorting_strategy=sorting_strategy,
                outer_n_folds=setup.outer_n_folds(),
                inner_n_folds=inner_n_folds,
                mutation=bitlist_mutation,
                use_clone_repurposing=setup.use_clone_repurposing(),
                adjuster_regressor=setup.adjuster_regressor())
        else:
            raise ValueError("Unknown MVMO algorithm.")

    mo_optimizer_by_fold = create_mo_optimizer_by_fold(
        mo_optimizer=mo_optimizer,
        feature_importance=feature_importance_by_fold,
        feature_selector=feature_selector_mo)

    return mo_optimizer_by_fold, input_data, objectives
