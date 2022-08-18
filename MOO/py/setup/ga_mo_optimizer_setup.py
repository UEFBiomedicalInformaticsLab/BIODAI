from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_star import NsgaStar
from cross_validation.multi_objective.optimizer.sweeping_strategy import SweepingStrategy
from folds_creator.input_data_k_folds_creator import InputDataKFoldsCreator
from ga_components.bitlist_mutation import FlipMutation, BitlistMutation
from ga_components.sorter.sorting_strategy import SortingStrategy, SortingStrategyCrowd
from hall_of_fame.population_observer_factory import ParetoFrontFactory, HofBySumFactory, HallOfFameFactory
from individual.num_features import NumFeatures, BinomialFromUniformNumFeatures, DEFAULT_INITIAL_FEATURES_MIN, \
    DEFAULT_INITIAL_FEATURES_MAX
from objective.social_objective import PersonalObjective

OUTER_N_FOLDS_BIG = 5
OUTER_N_FOLDS_SMALL = 2

POP_SMALL = 16
POP_BIG = 500
DEFAULT_MATING_PROB = 0.33
DEFAULT_MUTATING_FREQUENCY = 1.0
DEFAULT_INNER_N_FOLDS = 3
DEFAULT_USE_CLONE_REPURPOSING = False

CLASSIC_GENERATIONS_SMALL = 20
CLASSIC_GENERATIONS_BIG = 2000

DEFAULT_SWEEPING_STRATEGY_SMALL = SweepingStrategy([CLASSIC_GENERATIONS_SMALL])
DEFAULT_SWEEPING_STRATEGY_BIG = SweepingStrategy([CLASSIC_GENERATIONS_BIG])

DEFAULT_MUTATION_OPERATOR = FlipMutation()

DEFAULT_HOFS = (ParetoFrontFactory(), HofBySumFactory(size=50))

DEFAULT_INITIAL_FEATURES = BinomialFromUniformNumFeatures(
    min_num_features=DEFAULT_INITIAL_FEATURES_MIN,
    max_num_features=DEFAULT_INITIAL_FEATURES_MAX)

DEFAULT_CLASSIC_SORTING_STRATEGY = SortingStrategyCrowd()


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
