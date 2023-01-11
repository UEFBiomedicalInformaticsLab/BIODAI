from collections import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_star import NsgaStar
from cross_validation.multi_objective.optimizer.sweeping_ga_multi_objective_optimizer import \
    SweepingGAMultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.sweeping_strategy import SweepingStrategy
from feature_importance.multi_view_feature_importance import MVFeatureImportanceUniform
from folds_creator.input_data_k_folds_creator import InputDataKFoldsCreator
from ga_components.bitlist_mutation import FlipMutation, BitlistMutation
from ga_components.sorter.sorting_strategy import SortingStrategy, SortingStrategyCrowd, SortingStrategySocial
from hall_of_fame.population_observer_factory import ParetoFrontFactory, HallOfFameFactory, HofBySumFactory
from individual.num_features import BinomialFromUniformNumFeatures, NumFeatures, DEFAULT_INITIAL_FEATURES_MIN, \
    DEFAULT_INITIAL_FEATURES_MAX
from objective.social_objective import PersonalObjective
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

DEFAULT_SWEEPING_STRATEGY_SMALL = SweepingStrategy([GENERATIONS_PER_VIEW_SMALL]*SWEEPS_SMALL)
DEFAULT_SWEEPING_STRATEGY_BIG = SweepingStrategy([GENERATIONS_PER_VIEW_BIG]*SWEEPS_BIG)

DEFAULT_HOFS = (ParetoFrontFactory(), HofBySumFactory(size=50))

DEFAULT_MUTATION_OPERATOR = FlipMutation()

DEFAULT_INITIAL_FEATURES = BinomialFromUniformNumFeatures(
    min_num_features=DEFAULT_INITIAL_FEATURES_MIN,
    max_num_features=DEFAULT_INITIAL_FEATURES_MAX)


def sweeping_ga_mo_optimizer_setup(
        objectives: [PersonalObjective],
        printer: Printer = NullPrinter(),
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_freq: float = DEFAULT_MUTATING_FREQUENCY,
        sweeping_strategy: SweepingStrategy = DEFAULT_SWEEPING_STRATEGY_SMALL,
        sorting_strategy: SortingStrategy = DEFAULT_SWEEPING_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
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

    inner_folds_creator = InputDataKFoldsCreator(n_folds=inner_n_folds)

    return SweepingGAMultiObjectiveOptimizer(
        pop_size=pop_size, mutation_frequency=mutating_freq, mating_prob=mating_prob,
        initial_features=initial_features, folds_creator=inner_folds_creator,
        objectives=objectives,
        sorting_strategy=sorting_strategy,
        hof_factories=hofs,
        sweeping_strategy=sweeping_strategy,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing
    )


def small_sweeping_ga_mo_optimizer_setup(
        objectives: [PersonalObjective],
        sweeping_strategy: SweepingStrategy = DEFAULT_SWEEPING_STRATEGY_SMALL,
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
                                          verbose=verbose)


def big_sweeping_ga_mo_optimizer_setup(
        objectives: [PersonalObjective],
        printer: Printer = NullPrinter(),
        pop_size: int = POP_BIG,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        sweeping_strategy: SweepingStrategy = DEFAULT_SWEEPING_STRATEGY_BIG,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        sorting_strategy: SortingStrategy = DEFAULT_SWEEPING_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
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
                                          verbose=verbose)


def nsga_setup(
        objectives: [PersonalObjective],
        pop_size: int = POP_SMALL,
        initial_features: NumFeatures = DEFAULT_INITIAL_FEATURES,
        mating_prob: float = DEFAULT_MATING_PROB,
        mutating_prob: float = DEFAULT_MUTATING_FREQUENCY,
        n_gen: int = CLASSIC_GENERATIONS_SMALL,
        sorting_strategy: SortingStrategy = DEFAULT_CLASSIC_SORTING_STRATEGY,
        hofs: Sequence[HallOfFameFactory] = DEFAULT_HOFS,
        inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
        mutation: BitlistMutation = DEFAULT_MUTATION_OPERATOR,
        use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING
        ) -> MultiObjectiveOptimizerAcceptingFeatureImportance:

    inner_folds_creator = InputDataKFoldsCreator(n_folds=inner_n_folds)

    return NsgaStar(
        pop_size=pop_size, mutation_frequency=mutating_prob, mating_prob=mating_prob,
        n_gen=n_gen,
        initial_features=initial_features,
        folds_creator=inner_folds_creator,
        objectives=objectives,
        sorting_strategy=sorting_strategy,
        hof_factories=hofs,
        mutation=mutation,
        use_clone_repurposing=use_clone_repurposing
    )


def small_nsga_setup(
        objectives: [PersonalObjective],
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
                      hofs=hofs,
                      inner_n_folds=inner_n_folds,
                      mutation=mutation,
                      use_clone_repurposing=use_clone_repurposing)


def big_nsga_setup(
        objectives: [PersonalObjective],
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
                      hofs=hofs,
                      inner_n_folds=inner_n_folds,
                      mutation=mutation,
                      use_clone_repurposing=use_clone_repurposing)
