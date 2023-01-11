from collections import Iterable

from cross_validation.multi_objective.optimizer.mo_optimizer_including_feature_importance import \
    MOOptimizerIncludingFeatureImportance
from cross_validation.multi_objective.optimizer.sweeping_ga_multi_objective_optimizer import \
    SweepingGAMultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.sweeping_strategy import SweepingStrategy
from feature_importance.multi_outcome_feature_importance import MultiOutcomeFeatureImportance
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.num_features import NumFeatures
from objective.social_objective import PersonalObjective


class SweepingGAMOOptimizerFI(MOOptimizerIncludingFeatureImportance):

    def __init__(self, pop_size, mutation_frequency, mating_prob,
                 sweeping_strategy: SweepingStrategy,
                 initial_features: NumFeatures,
                 folds_creator: InputDataFoldsCreator,
                 objectives: Iterable[PersonalObjective],
                 sorting_strategy: SortingStrategy,
                 mv_feature_importance: MultiOutcomeFeatureImportance,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 mutation: BitlistMutation = FlipMutation(),
                 use_clone_repurposing: bool = False):
        MOOptimizerIncludingFeatureImportance.__init__(
            self, feature_importance=mv_feature_importance,
            optimizer=SweepingGAMultiObjectiveOptimizer(
                pop_size=pop_size, mutation_frequency=mutation_frequency, mating_prob=mating_prob,
                initial_features=initial_features, folds_creator=folds_creator,
                objectives=objectives,
                sweeping_strategy=sweeping_strategy,
                sorting_strategy=sorting_strategy,
                hof_factories=hof_factories,
                mutation=mutation,
                use_clone_repurposing=use_clone_repurposing
            ))
