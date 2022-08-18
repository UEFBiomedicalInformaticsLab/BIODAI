from cross_validation.multi_objective.optimizer.composite_mo_optimizer_by_fold import \
    CompositeMultiObjectiveOptimizerByFold
from cross_validation.multi_objective.optimizer.lasso_mo import LassoMO
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold,\
    DummyMultiObjectiveOptimizerByFold
from cross_validation.multi_objective.optimizer.so_to_mo_optimizer_adapter import SOtoMOOptimizerAdapter
from cross_validation.single_objective.optimizer.lasso_optimizer import LassoSingleObjectiveOptimizer
from ga_components.bitlist_mutation import FlipMutation, SymmetricFlipMutation
from ga_components.selection import ElitistSelection
from ga_components.sorter.sorting_strategy import SortingStrategyCrowd, SortingStrategyCrowdFull, SortingStrategyCrowdCI
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from load_omics_views import MRNA_NAME
from objective.social_objective import PersonalObjective
from setup.allowed_names import CROWDING_DISTANCE_NAME, CROWDING_DISTANCE_FULL_NAME, CROWDING_DISTANCE_CI_NAME, \
    LASSO_NAME, NSGA_STAR_NAME, LASSO_MO_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import big_nsga_setup, small_nsga_setup
from setup.parse_feature_importance import parse_feature_importance_by_fold
from setup.parse_initial_features import parse_initial_features
from setup.parse_objectives import parse_objectives
from setup.setup_utils import load_input_data
from util.printer.printer import Printer


def setup_to_mo_optimizer(setup: EvaluationSetup, printer: Printer
                          ) -> tuple[MultiObjectiveOptimizerByFold, InputData, [PersonalObjective]]:

    dataset_name = setup.dataset()
    mvmo_algorithm = setup.mvmo_algorithm()
    use_big_setup = setup.use_big_defaults()
    mating_prob = setup.mating_prob()
    mutation_frequency = setup.mutation_frequency()
    sorting_strategy_str = setup.sorting_strategy()
    generations = setup.generations()
    views_to_use = [MRNA_NAME]
    pop = setup.pop()
    inner_n_folds = setup.inner_n_folds()

    selection = ElitistSelection()

    input_data = load_input_data(dataset_name=dataset_name, views_to_use=views_to_use, printer=printer)

    if mvmo_algorithm == LASSO_NAME or mvmo_algorithm == LASSO_MO_NAME:
        use_inner_model = False
    elif mvmo_algorithm == NSGA_STAR_NAME:
        use_inner_model = True
    else:
        raise ValueError("Unknown MVMO algorithm.")

    objectives = parse_objectives(
        objectives_str=setup.objectives(),
        default_target=input_data.stratify_outcome_name(),
        use_model=use_inner_model,
        logistic_max_iter=setup.logistic_max_iter())
    printer.print_variable("Objectives", objectives)

    n_objectives = len(objectives)

    if sorting_strategy_str == CROWDING_DISTANCE_NAME:
        sorting_strategy = SortingStrategyCrowd(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_FULL_NAME:
        sorting_strategy = SortingStrategyCrowdFull(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_CI_NAME:
        sorting_strategy = SortingStrategyCrowdCI(selection=selection)
    else:
        raise ValueError("Unknown sorting strategy.")

    feature_importance = parse_feature_importance_by_fold(
        categorical_fi_str=setup.feature_importance_categorical())

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

    run_nsga = (mvmo_algorithm == NSGA_STAR_NAME)

    if mvmo_algorithm == LASSO_NAME:
        mo_optimizer = SOtoMOOptimizerAdapter(
            so_optimizer=LassoSingleObjectiveOptimizer(), n_objectives=n_objectives)
    elif mvmo_algorithm == LASSO_MO_NAME:
        if use_big_setup:
            mo_optimizer = LassoMO(n_objectives=n_objectives, shrink_factor=0.99)
        else:
            mo_optimizer = LassoMO(n_objectives=n_objectives)
    else:
        if use_big_setup:
            if run_nsga:
                mo_optimizer = big_nsga_setup(
                    objectives=objectives,
                    pop_size=pop,
                    initial_features=initial_features,
                    n_gen=sum(generations),
                    mating_prob=mating_prob, mutating_prob=mutation_frequency,
                    sorting_strategy=sorting_strategy,
                    inner_n_folds=inner_n_folds,
                    mutation=bitlist_mutation,
                    use_clone_repurposing=setup.use_clone_repurposing())
            else:
                raise ValueError("Unknown MVMO algorithm.")
        else:
            if run_nsga:
                mo_optimizer = small_nsga_setup(
                    objectives=objectives,
                    pop_size=pop,
                    initial_features=initial_features,
                    n_gen=sum(generations),
                    mating_prob=mating_prob, mutating_prob=mutation_frequency,
                    sorting_strategy=sorting_strategy,
                    inner_n_folds=inner_n_folds,
                    mutation=bitlist_mutation,
                    use_clone_repurposing=setup.use_clone_repurposing())
            else:
                raise ValueError("Unknown MVMO algorithm.")
    if isinstance(mo_optimizer, MultiObjectiveOptimizerAcceptingFeatureImportance):
        mo_optimizer_by_fold = CompositeMultiObjectiveOptimizerByFold(
            fi=feature_importance,
            optimizer=mo_optimizer)
    else:
        mo_optimizer_by_fold = DummyMultiObjectiveOptimizerByFold(optimizer=mo_optimizer)

    return mo_optimizer_by_fold, input_data, objectives
