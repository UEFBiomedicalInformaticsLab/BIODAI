from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.adjusted_optimizer import AdjustedOptimizer, \
    TUNING_HOFS, ScaleParameters, tuning_parameters, \
    DEFAULT_ADJUSTER_REGRESSOR
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_star import NsgaStar
from cross_validation.multi_objective.optimizer.nsga.nsga_star_factory import NsgaStarFactory
from cross_validation.multi_objective.optimizer.pso.cmdpsofs import CMDPSOFS
from cross_validation.multi_objective.optimizer.sweeping_ga_multi_objective_optimizer import \
    SweepingGAMultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from feature_importance.multi_view_feature_importance import MVFeatureImportanceUniform
from fitness_adjuster.fitness_adjuster_learner import FitnessAdjusterLearner
from folds_creator.default_folds_creator import default_folds_creator
from ga_components.bitlist_mutation import FlipMutation, BitlistMutation
from ga_components.sorter.sorting_strategy import SortingStrategy, SortingStrategyCrowd, SortingStrategySocial
from hall_of_fame.population_observer_factory import ParetoFrontFactory, HallOfFameFactory, HofBySumFactory, \
    LastPopFactory
from individual.num_features import BinomialFromUniformNumFeatures, NumFeatures, DEFAULT_INITIAL_FEATURES_MIN, \
    DEFAULT_INITIAL_FEATURES_MAX
from model.regression.svregressor import RegressorSVModel
from objective.social_objective import PersonalObjective
from run_ga.master_runner import MasterRunner, ResamplingMaster
from util.printer.printer import Printer, NullPrinter


OUTER_N_FOLDS_BIG = 5
OUTER_N_FOLDS_SMALL = 2

POP_SMALL = 16
POP_BIG = 500
GENERATIONS_PER_VIEW_SMALL = 5
GENERATIONS_PER_VIEW_BIG = 200
SWEEPS_SMALL = 2
SWEEPS_BIG = 5
DEFAULT_MATING_PROB = 0.33
DEFAULT_MUTATING_FREQUENCY = 1.0
DEFAULT_INNER_N_FOLDS = 3
DEFAULT_SWEEPING_SORTING_STRATEGY = SortingStrategySocial()
DEFAULT_CLASSIC_SORTING_STRATEGY = SortingStrategyCrowd()
DEFAULT_FEATURE_IMPORTANCE = MVFeatureImportanceUniform()
DEFAULT_USE_CLONE_REPURPOSING = False

CLASSIC_GENERATIONS_SMALL = GENERATIONS_PER_VIEW_SMALL*SWEEPS_SMALL*2
CLASSIC_GENERATIONS_BIG = GENERATIONS_PER_VIEW_BIG*SWEEPS_BIG*2

DEFAULT_SWEEPING_STRATEGY_SMALL = GenerationsStrategy([GENERATIONS_PER_VIEW_SMALL] * SWEEPS_SMALL)
DEFAULT_SWEEPING_STRATEGY_BIG = GenerationsStrategy([GENERATIONS_PER_VIEW_BIG] * SWEEPS_BIG)

DEFAULT_HOFS = (ParetoFrontFactory(), LastPopFactory(), HofBySumFactory(size=50), HofBySumFactory(size=100))

DEFAULT_MUTATION_OPERATOR = FlipMutation()

DEFAULT_INITIAL_FEATURES = BinomialFromUniformNumFeatures(
    min_num_features=DEFAULT_INITIAL_FEATURES_MIN,
    max_num_features=DEFAULT_INITIAL_FEATURES_MAX)


def sweeping_ga_mo_optimizer_setup(
        objectives: Sequence[PersonalObjective],
        printer: Printer = NullPrinter(),
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_freq: float = DEFAULT_MUTATING_FREQUENCY,
        sweeping_strategy: GenerationsStrategy = DEFAULT_SWEEPING_STRATEGY_SMALL,
        sorting_strategy: SortingStrategy = DEFAULT_SWEEPING_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        master_runner: MasterRunner = ResamplingMaster(),
        verbose=False) -> SweepingGAMultiObjectiveOptimizer:

    if verbose:
        printer.title_print("Setting up multi-objective GA main parameters")
        printer.print_variable("Objectives", objectives)
        printer.print_variable("Population size", pop_size)
        printer.print_variable("Initial features", initial_features)
        printer.print_variable("Number of generations per sweep", sweeping_strategy)
        printer.print_variable("Crossover probability", mating_prob)
        printer.print_variable("Mutation probability", mutating_freq)
        printer.print_variable("Number of inner folds", inner_n_folds)
        printer.print_variable("Sorting strategy", sorting_strategy)
        printer.print_variable("Use clone repurposing", use_clone_repurposing)
        printer.print_variable("Mutation operator", mutation)

    inner_folds_creator = default_folds_creator(n_folds=inner_n_folds)

    return SweepingGAMultiObjectiveOptimizer(
        pop_size=pop_size, mutation_frequency=mutating_freq, mating_prob=mating_prob,
        initial_features=initial_features, folds_creator=inner_folds_creator,
        objectives=objectives,
        sorting_strategy=sorting_strategy,
        hof_factories=hofs,
        sweeping_strategy=sweeping_strategy,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing,
        master_runner=master_runner
    )


def small_sweeping_ga_mo_optimizer_setup(
        objectives: Sequence[PersonalObjective],
        sweeping_strategy: GenerationsStrategy = DEFAULT_SWEEPING_STRATEGY_SMALL,
        printer: Printer = NullPrinter(),
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        sorting_strategy: SortingStrategy = DEFAULT_SWEEPING_SORTING_STRATEGY,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        master_runner: MasterRunner = ResamplingMaster(),
        verbose=False) -> SweepingGAMultiObjectiveOptimizer:
    return sweeping_ga_mo_optimizer_setup(objectives=objectives,
                                          sweeping_strategy=sweeping_strategy,
                                          printer=printer,
                                          pop_size=pop_size,
                                          initial_features=initial_features,
                                          mating_prob=mating_prob,
                                          mutating_freq=mutating_prob,
                                          sorting_strategy=sorting_strategy,
                                          hofs=hofs,
                                          inner_n_folds=inner_n_folds,
                                          mutation=mutation,
                                          use_clone_repurposing=use_clone_repurposing,
                                          master_runner=master_runner,
                                          verbose=verbose)


def big_sweeping_ga_mo_optimizer_setup(
        objectives: Sequence[PersonalObjective],
        printer: Printer = NullPrinter(),
        pop_size: int = POP_BIG,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        sweeping_strategy: GenerationsStrategy = DEFAULT_SWEEPING_STRATEGY_BIG,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_SWEEPING_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        master_runner: MasterRunner = ResamplingMaster(),
        verbose=False) -> SweepingGAMultiObjectiveOptimizer:
    return sweeping_ga_mo_optimizer_setup(objectives=objectives,
                                          printer=printer,
                                          pop_size=pop_size,
                                          initial_features=initial_features,
                                          mating_prob=mating_prob,
                                          mutating_freq=mutating_prob,
                                          sweeping_strategy=sweeping_strategy,
                                          sorting_strategy=sorting_strategy,
                                          hofs=hofs,
                                          inner_n_folds=inner_n_folds,
                                          mutation=mutation,
                                          use_clone_repurposing=use_clone_repurposing,
                                          master_runner=master_runner,
                                          verbose=verbose)


def nsga_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hof_factories: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:

    inner_folds_creator = default_folds_creator(n_folds=inner_n_folds)

    return NsgaStar(
        pop_size=pop_size, mutation_frequency=mutating_prob, mating_prob=mating_prob,
        n_gen=n_gen,
        initial_features=initial_features,
        folds_creator=inner_folds_creator,
        objectives=objectives,
        sorting_strategy=sorting_strategy,
        hof_factories=hof_factories,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing
    )


def pso_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        hof_factories: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:

    inner_folds_creator = default_folds_creator(n_folds=inner_n_folds)

    return CMDPSOFS(
        pop_size=pop_size,
        initial_features=initial_features,
        n_gen=n_gen,
        folds_creator=inner_folds_creator,
        objectives=objectives,
        hof_factories=hof_factories
    )


def small_nsga_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return nsga_setup(objectives=objectives, pop_size=pop_size,
                      initial_features=initial_features,
                      mating_prob=mating_prob,
                      mutating_prob=mutating_prob,
                      n_gen=n_gen,
                      sorting_strategy=sorting_strategy,
                      hof_factories=hofs,
                      inner_n_folds=inner_n_folds,
                      mutation=mutation,
                      use_clone_repurposing=use_clone_repurposing)


def small_pso_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return pso_setup(objectives=objectives, pop_size=pop_size,
                     initial_features=initial_features,
                     n_gen=n_gen,
                     hof_factories=hofs,
                     inner_n_folds=inner_n_folds)


def big_nsga_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_BIG,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_BIG,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return nsga_setup(objectives=objectives, pop_size=pop_size,
                      n_gen=n_gen,
                      initial_features=initial_features,
                      mating_prob=mating_prob,
                      mutating_prob=mutating_prob,
                      sorting_strategy=sorting_strategy,
                      hof_factories=hofs,
                      inner_n_folds=inner_n_folds,
                      mutation=mutation,
                      use_clone_repurposing=use_clone_repurposing)


def big_pso_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_BIG,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_BIG,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return pso_setup(objectives=objectives, pop_size=pop_size,
                     initial_features=initial_features,
                     n_gen=n_gen,
                     hof_factories=hofs,
                     inner_n_folds=inner_n_folds)


def adjusted_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutation_frequency: float = DEFAULT_MUTATING_FREQUENCY,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hof_factories: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        outer_n_folds: int = OUTER_N_FOLDS_SMALL,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        adjuster_regressor: RegressorSVModel = DEFAULT_ADJUSTER_REGRESSOR
) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    main_parameters = ScaleParameters(
        pop_size=pop_size, n_gen=n_gen, n_folds=outer_n_folds, inner_n_folds=inner_n_folds)
    tuning_scale_parameters = tuning_parameters(main_parameters=main_parameters)
    main_inner_folds_creator = default_folds_creator(n_folds=inner_n_folds)
    tuning_folds_creator = default_folds_creator(n_folds=tuning_scale_parameters.n_folds)
    tuning_optimizer = nsga_setup(
        objectives=objectives,
        pop_size=tuning_scale_parameters.pop_size,
        n_gen=tuning_scale_parameters.n_gen,
        initial_features=initial_features,
        mating_prob=mating_prob,
        mutating_prob=mutation_frequency,
        sorting_strategy=sorting_strategy,
        hof_factories=TUNING_HOFS,
        inner_n_folds=tuning_scale_parameters.inner_n_folds,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing
    )
    adjuster_learner = FitnessAdjusterLearner(model=adjuster_regressor)
    main_optimizer_factory = NsgaStarFactory(
        pop_size=pop_size,
        n_gen=n_gen,
        initial_features=initial_features,
        mating_prob=mating_prob,
        mutation_frequency=mutation_frequency,
        folds_creator=main_inner_folds_creator,
        sorting_strategy=sorting_strategy,
        hof_factories=hof_factories,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing)
    return AdjustedOptimizer(
        tuning_folds_creator=tuning_folds_creator,
        objectives=objectives,
        tuning_optimizer=tuning_optimizer,
        adjuster_learner=adjuster_learner,
        main_optimizer_factory=main_optimizer_factory
    )


def big_adjusted_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_BIG,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_BIG,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutation_frequency: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hof_factories: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        outer_n_folds: int = OUTER_N_FOLDS_BIG,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        adjuster_regressor: RegressorSVModel = DEFAULT_ADJUSTER_REGRESSOR
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return adjusted_setup(objectives=objectives, pop_size=pop_size,
                          n_gen=n_gen,
                          initial_features=initial_features,
                          mating_prob=mating_prob,
                          mutation_frequency=mutation_frequency,
                          sorting_strategy=sorting_strategy,
                          hof_factories=hof_factories,
                          outer_n_folds=outer_n_folds,
                          inner_n_folds=inner_n_folds,
                          mutation=mutation,
                          use_clone_repurposing=use_clone_repurposing,
                          adjuster_regressor=adjuster_regressor)


def small_adjusted_setup(
        objectives: Sequence[PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutation_frequency: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        outer_n_folds: int = OUTER_N_FOLDS_SMALL,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
        adjuster_regressor: RegressorSVModel = DEFAULT_ADJUSTER_REGRESSOR
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
    return adjusted_setup(
        objectives=objectives, pop_size=pop_size,
        initial_features=initial_features,
        mating_prob=mating_prob,
        mutation_frequency=mutation_frequency,
        n_gen=n_gen,
        sorting_strategy=sorting_strategy,
        hof_factories=hofs,
        outer_n_folds=outer_n_folds,
        inner_n_folds=inner_n_folds,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing,
        adjuster_regressor=adjuster_regressor)
