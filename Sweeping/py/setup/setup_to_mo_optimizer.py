from collections.abc import Sequence
from typing import Optional

from cross_validation.multi_objective.optimizer.guided_forward_accepting_fi import GuidedForwardAcceptingFI
from cross_validation.multi_objective.optimizer.lasso_mo import LassoMO
from cross_validation.multi_objective.optimizer.mo_optimizer_factory import create_mo_optimizer_by_fold
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold
from cross_validation.multi_objective.optimizer.pso.cmdpsofs import CMDPSOFS_NICK
from cross_validation.multi_objective.optimizer.rfe_mo_optimizer import RfeMoOptimizer
from cross_validation.multi_objective.optimizer.so_to_mo_optimizer_adapter import SOtoMOOptimizerAdapter
from cross_validation.multi_objective.optimizer.user_provided.pam50 import Pam50
from cross_validation.single_objective.optimizer.lasso_optimizer import LassoSingleObjectiveOptimizer
from folds_creator.default_folds_creator import default_folds_creator
from ga_components.bitlist_mutation import FlipMutation, SymmetricFlipMutation, PersonalizedMutation
from ga_components.selection import ElitistSelection, TournamentExtraction
from ga_components.sorter.sorting_strategy import SortingStrategySocial, SortingStrategyCrowd, \
    SortingStrategySocialFull, SortingStrategyCrowdFull, SortingStrategyCrowdCI, SortingStrategySocialCI, \
    SortingStrategyNsga3, SortingStrategyNsga3CI
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from run_ga.master_runner import ResamplingMaster, FatConcatenatedMaster, LeanConcatenatedMaster
from setup.allowed_names import SOCIAL_SPACE_NAME, CROWDING_DISTANCE_NAME, SOCIAL_SPACE_FULL_NAME, \
    CROWDING_DISTANCE_FULL_NAME, CROWDING_DISTANCE_CI_NAME, SOCIAL_SPACE_CI_NAME, LASSO_NAME, NSGA_STAR_NAME, \
    RESAMPLED_SWEEPING_NAME, LASSO_MO_NAME, GUIDED_FORWARD_NAME, RFE_NAME, NSGA3_NAME, PAM50_NAME, NSGA3_CI_NAME, \
    FAT_CONCATENATED_SWEEPING_NAME, LEAN_CONCATENATED_SWEEPING_NAME, ADJUSTED_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import big_nsga_setup, big_sweeping_ga_mo_optimizer_setup, \
    small_nsga_setup, small_sweeping_ga_mo_optimizer_setup, DEFAULT_HOFS, big_adjusted_setup, small_adjusted_setup, \
    big_pso_setup, small_pso_setup
from setup.parse_feature_importance import parse_feature_importance_by_fold
from setup.parse_initial_features import parse_initial_features
from setup.parse_objectives import parse_objectives
from setup.setup_utils import load_input_data
from univariate_feature_selection.many_feature_selector import AnovaAndCoxManyFeatureSelector, ManyFeatureSelector, \
    DummyManyFeatureSelector, CompositeManyFeatureSelector, \
    ManyFeatureSelectorCox
from univariate_feature_selection.univariate_feature_selector_descriptor import \
    DUMMY_SELECTOR_MANY_DESCRIPTOR, \
    ManyFeatureSelectorPipelineDescriptor, ManyFeatureSelectorDescriptor, \
    FdrManyFeatureSelectorClassDescriptor, ManyFeatureSelectorFromSingleDescriptor, HWESingleFeatureSelectorDescriptor, \
    ANOVA_CATEGORICAL_DESCRIPTOR, MinorFrequencySingleFeatureSelectorDescriptor, MAFSingleFeatureSelectorDescriptor
from univariate_feature_selection.feature_selector_multi_target import DummySelectorMO, FeatureSelectorMOUnion
from univariate_feature_selection.univariate_feature_selector_generators import DEFAULT_UNIVARIATE_FS_GENERATOR
from univariate_property_computer.univariate_property_computer_descriptor import \
    LOG_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR, ANOVA_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR
from util.printer.printer import Printer
from util.recursive_feature_importance import MultiOutcomeRecursiveFeatureImportanceLasso


UNIVARIATE_FS_CATEGORICAL_DESCRIPTORS_NO_FDR = [
    DUMMY_SELECTOR_MANY_DESCRIPTOR,
    ANOVA_CATEGORICAL_DESCRIPTOR,
    ManyFeatureSelectorFromSingleDescriptor(single_fs=HWESingleFeatureSelectorDescriptor()),
    ManyFeatureSelectorFromSingleDescriptor(single_fs=MinorFrequencySingleFeatureSelectorDescriptor()),
    ManyFeatureSelectorFromSingleDescriptor(single_fs=MAFSingleFeatureSelectorDescriptor())
]

UNIVARIATE_PVAL_COMPUTER_CATEGORICAL_DESCRIPTORS = [
    ANOVA_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR,
    LOG_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR
]


def univariate_fs_categorical_descriptor(
        fs_str: str,
        fdr: Optional[float]) -> ManyFeatureSelectorDescriptor:
    for desc in UNIVARIATE_FS_CATEGORICAL_DESCRIPTORS_NO_FDR:
        if fs_str == desc.algorithm_nick():
            return desc
    for desc in UNIVARIATE_PVAL_COMPUTER_CATEGORICAL_DESCRIPTORS:
        if fs_str == FdrManyFeatureSelectorClassDescriptor(computer=desc).algorithm_nick():
            return FdrManyFeatureSelectorClassDescriptor(computer=desc, fdr_threshold=fdr)
    raise ValueError("Unknown univariate categorical feature selector: " + fs_str)


def univariate_fs_categorical_pipe_descriptor(
        categorical_fs_str: Sequence[str],
        fdr: Optional[float]) -> ManyFeatureSelectorDescriptor:
    fs_in_pipe = [univariate_fs_categorical_descriptor(fs_str=fs_str, fdr=fdr) for fs_str in categorical_fs_str]
    return ManyFeatureSelectorPipelineDescriptor(selectors=fs_in_pipe)


def parse_univariate_fs_categorical(
        categorical_fs_str: Optional[Sequence[str]],
        fdr: Optional[float]) -> Optional[ManyFeatureSelector]:
    if categorical_fs_str is None:
        return None
    else:
        descriptor = univariate_fs_categorical_pipe_descriptor(categorical_fs_str=categorical_fs_str, fdr=fdr)
        return DEFAULT_UNIVARIATE_FS_GENERATOR.generate(descriptor=descriptor)


def uses_univariate_fs(mvmo_algorithm: str) -> bool:
    return not (mvmo_algorithm == LASSO_NAME
                or mvmo_algorithm == LASSO_MO_NAME
                or mvmo_algorithm == GUIDED_FORWARD_NAME
                or mvmo_algorithm == RFE_NAME
                or mvmo_algorithm == PAM50_NAME)


def setup_to_mo_optimizer(setup: EvaluationSetup, printer: Printer
                          ) -> tuple[MultiObjectiveOptimizerByFold, InputData, list[PersonalObjectiveWithImportance]]:

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

    use_resampling = (mvmo_algorithm == RESAMPLED_SWEEPING_NAME)
    if use_resampling:
        master_runner = ResamplingMaster()
    else:
        if mvmo_algorithm == LEAN_CONCATENATED_SWEEPING_NAME:
            master_runner = LeanConcatenatedMaster()
        else:
            master_runner = FatConcatenatedMaster()

    selection_str = setup.selection()
    if selection_str == ElitistSelection().name():
        selection = ElitistSelection()
    elif selection_str == TournamentExtraction.base_nick():
        selection = TournamentExtraction(n_participants=setup.selection_tournament_size())
    else:
        raise ValueError("Unknown selection: " + selection_str)

    skip_plotting_huge_views = not setup.draw_huge_views()
    input_data = load_input_data(dataset_name=dataset_name, views_to_use=views_to_use, printer=printer,
                                 covariate_views=setup.univariate_fs_covariates(),
                                 skip_plotting_huge_views=skip_plotting_huge_views)

    run_nsga = (mvmo_algorithm == NSGA_STAR_NAME)
    run_pso = (mvmo_algorithm == CMDPSOFS_NICK)
    run_sweeping_ga = (mvmo_algorithm == RESAMPLED_SWEEPING_NAME or mvmo_algorithm == FAT_CONCATENATED_SWEEPING_NAME or
                       mvmo_algorithm == LEAN_CONCATENATED_SWEEPING_NAME)
    run_adjusted = mvmo_algorithm == ADJUSTED_NAME

    if mvmo_algorithm == LASSO_NAME or mvmo_algorithm == LASSO_MO_NAME:
        use_inner_model = False
    elif (run_nsga or
          run_sweeping_ga or
          mvmo_algorithm == GUIDED_FORWARD_NAME or mvmo_algorithm == RFE_NAME or mvmo_algorithm == PAM50_NAME or
          run_adjusted or run_pso):
        use_inner_model = True
    else:
        raise ValueError("Unknown MVMO algorithm.")

    objectives = parse_objectives(
        objectives_str=setup.objectives(),
        default_target=input_data.stratify_outcome_name(),
        use_model=use_inner_model,
        max_sd=setup.max_deviation(),
        logistic_max_iter=setup.logistic_max_iter(),
        penalty=setup.penalty(),
        outcomes=input_data.outcomes())
    printer.print_variable("Objectives", objectives)

    n_objectives = len(objectives)

    if sorting_strategy_str == SOCIAL_SPACE_NAME:
        sorting_strategy = SortingStrategySocial(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_NAME:
        sorting_strategy = SortingStrategyCrowd(selection=selection)
    elif sorting_strategy_str == SOCIAL_SPACE_FULL_NAME:
        sorting_strategy = SortingStrategySocialFull(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_FULL_NAME:
        sorting_strategy = SortingStrategyCrowdFull(selection=selection)
    elif sorting_strategy_str == CROWDING_DISTANCE_CI_NAME:
        sorting_strategy = SortingStrategyCrowdCI(selection=selection)
    elif sorting_strategy_str == SOCIAL_SPACE_CI_NAME:
        sorting_strategy = SortingStrategySocialCI(selection=selection)
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
    elif bitlist_mutation_str == PersonalizedMutation().nick():
        bitlist_mutation = PersonalizedMutation()
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

    univariate_fs_categorical = parse_univariate_fs_categorical(
        categorical_fs_str=setup.univariate_fs_categorical(),
        fdr=setup.univariate_fs_fdr())
    if uses_univariate_fs(mvmo_algorithm=mvmo_algorithm):
        if univariate_fs_categorical is None:
            feature_selector_mo = FeatureSelectorMOUnion(feature_selector_so=AnovaAndCoxManyFeatureSelector())
        else:
            feature_selector_mo = FeatureSelectorMOUnion(
                feature_selector_so=CompositeManyFeatureSelector(
                    categorical_selector=univariate_fs_categorical, survival_selector=ManyFeatureSelectorCox()))
    else:
        if univariate_fs_categorical is None:
            feature_selector_mo = DummySelectorMO()
        else:
            feature_selector_mo = FeatureSelectorMOUnion(
                feature_selector_so=CompositeManyFeatureSelector(
                    categorical_selector=univariate_fs_categorical, survival_selector=DummyManyFeatureSelector()))

    if mvmo_algorithm == LASSO_NAME:
        mo_optimizer = SOtoMOOptimizerAdapter(
            so_optimizer=LassoSingleObjectiveOptimizer(), n_objectives=n_objectives)
    elif mvmo_algorithm == LASSO_MO_NAME:
        if use_big_setup:
            mo_optimizer = LassoMO(objectives=objectives, shrink_factor=0.99)
        else:
            mo_optimizer = LassoMO(objectives=objectives)
    elif mvmo_algorithm == GUIDED_FORWARD_NAME:
        mo_optimizer = GuidedForwardAcceptingFI(
            folds_creator=default_folds_creator(n_folds=inner_n_folds),
            objectives=objectives,
            hof_factories=DEFAULT_HOFS
            )
    elif mvmo_algorithm == RFE_NAME:
        mo_optimizer = RfeMoOptimizer(
            objectives=objectives,
            recursive_fi=MultiOutcomeRecursiveFeatureImportanceLasso(),
            folds_creator=default_folds_creator(n_folds=inner_n_folds),
            hof_factories=DEFAULT_HOFS)
    elif mvmo_algorithm == PAM50_NAME:
        mo_optimizer = Pam50(
            objectives=objectives,
            folds_creator=default_folds_creator(n_folds=inner_n_folds),
            hof_factories=DEFAULT_HOFS)
    else:  # Using GAs or PSOs
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
            elif run_sweeping_ga:
                mo_optimizer = big_sweeping_ga_mo_optimizer_setup(
                    objectives=objectives, sweeping_strategy=generations,
                    mating_prob=mating_prob, mutating_prob=mutation_frequency,
                    pop_size=pop,
                    initial_features=initial_features,
                    sorting_strategy=sorting_strategy,
                    inner_n_folds=inner_n_folds,
                    printer=printer,
                    mutation=bitlist_mutation,
                    use_clone_repurposing=setup.use_clone_repurposing(),
                    master_runner=master_runner)
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
            elif run_pso:
                mo_optimizer = big_pso_setup(
                    objectives=objectives,
                    pop_size=pop,
                    initial_features=initial_features,
                    n_gen=generations.concatenated_generations(),
                    inner_n_folds=inner_n_folds)
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
            elif run_sweeping_ga:
                mo_optimizer = small_sweeping_ga_mo_optimizer_setup(
                    objectives=objectives,
                    sweeping_strategy=generations,
                    pop_size=pop,
                    initial_features=initial_features,
                    mating_prob=mating_prob, mutating_prob=mutation_frequency,
                    sorting_strategy=sorting_strategy,
                    inner_n_folds=inner_n_folds,
                    mutation=bitlist_mutation,
                    use_clone_repurposing=setup.use_clone_repurposing(),
                    master_runner=master_runner,
                    printer=printer)
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
            elif run_pso:
                mo_optimizer = small_pso_setup(
                    objectives=objectives,
                    pop_size=pop,
                    initial_features=initial_features,
                    n_gen=generations.concatenated_generations(),
                    inner_n_folds=inner_n_folds)
            else:
                raise ValueError("Unknown MVMO algorithm.")

    mo_optimizer_by_fold = create_mo_optimizer_by_fold(
        mo_optimizer=mo_optimizer,
        feature_importance=feature_importance_by_fold,
        feature_selector=feature_selector_mo)

    return mo_optimizer_by_fold, input_data, objectives
