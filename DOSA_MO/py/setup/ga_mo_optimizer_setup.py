from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.adjusted_optimizer import AdjustedOptimizer, \
    TUNING_HOFS, ScaleParameters, tuning_parameters, \
    DEFAULT_ADJUSTER_REGRESSOR
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_star import NsgaStar
from cross_validation.multi_objective.optimizer.nsga.nsga_star_factory import NsgaStarFactory
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from feature_importance.multi_view_feature_importance import MVFeatureImportanceUniform
from fitness_adjuster.fitness_adjuster_learner import FitnessAdjusterLearner
from folds_creator.default_folds_creator import default_folds_creator
from ga_components.bitlist_mutation import FlipMutation, BitlistMutation
from ga_components.sorter.sorting_strategy import SortingStrategy, SortingStrategyCrowd
from hall_of_fame.population_observer_factory import ParetoFrontFactory, HallOfFameFactory, HofBySumFactory, \
    LastPopFactory
from individual.num_features import BinomialFromUniformNumFeatures, NumFeatures, DEFAULT_INITIAL_FEATURES_MIN, \
    DEFAULT_INITIAL_FEATURES_MAX
from model.regression.regressor import RegressorModel
from objective.social_objective import PersonalObjective


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


def nsga_setup(
        objectives: [PersonalObjective],
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
                      hof_factories=hofs,
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
                      hof_factories=hofs,
                      inner_n_folds=inner_n_folds,
                      mutation=mutation,
                      use_clone_repurposing=use_clone_repurposing)


def adjusted_setup(
        objectives: [PersonalObjective],
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
        adjuster_regressor: RegressorModel = DEFAULT_ADJUSTER_REGRESSOR
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
        objectives: [PersonalObjective],
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
        adjuster_regressor: RegressorModel = DEFAULT_ADJUSTER_REGRESSOR
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
        objectives: [PersonalObjective],
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
        adjuster_regressor: RegressorModel = DEFAULT_ADJUSTER_REGRESSOR
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
