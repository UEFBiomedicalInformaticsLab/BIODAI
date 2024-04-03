from collections.abc import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.mo_optimizer_factory import MOOptimizerFactory
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_star import NsgaStar, nsga_nick, nsga_name
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.num_features import NumFeatures
from objective.social_objective import PersonalObjective
from util.utils import name_value


class NsgaStarFactory(MOOptimizerFactory):
    __sorting_strategy: SortingStrategy
    __hof_factories: Iterable[HallOfFameFactory]
    __nick: str
    __name: str
    __folds_creator: InputDataFoldsCreator
    __mutation: BitlistMutation
    __initial_features: NumFeatures
    __use_clone_repurposing: bool

    def __init__(self, pop_size, mutation_frequency, mating_prob, n_gen,
                 initial_features: NumFeatures,
                 folds_creator: InputDataFoldsCreator,
                 sorting_strategy: SortingStrategy,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 mutation: BitlistMutation = FlipMutation(),
                 use_clone_repurposing: bool = False):
        self.__pop_size = pop_size
        self.__mutation_frequency = mutation_frequency
        self.__mating_prob = mating_prob
        self.__n_gen = n_gen
        self.__initial_features = initial_features
        self.__folds_creator = folds_creator
        self.__sorting_strategy = sorting_strategy
        self.__hof_factories = hof_factories
        self.__mutation = mutation
        self.__use_clone_repurposing = use_clone_repurposing
        self.__nick = nsga_nick(
            sorting_strategy=sorting_strategy,
            folds_creator=folds_creator,
            pop_size=pop_size,
            initial_features=initial_features,
            n_gen=n_gen,
            use_clone_repurposing=use_clone_repurposing,
            mating_prob=mating_prob,
            mutation=mutation,
            mutation_frequency=mutation_frequency)
        self.__name = nsga_name(
            sorting_strategy=sorting_strategy,
            folds_creator=folds_creator,
            pop_size=pop_size,
            initial_features=initial_features,
            n_gen=n_gen,
            use_clone_repurposing=use_clone_repurposing,
            mating_prob=mating_prob,
            mutation=mutation,
            mutation_frequency=mutation_frequency)

    def create_optimizer(self,
                         objectives: Sequence[PersonalObjective]) -> MultiObjectiveOptimizerAcceptingFeatureImportance:
        return NsgaStar(
            pop_size=self.__pop_size, mutation_frequency=self.__mutation_frequency,
            mating_prob=self.__mating_prob, n_gen=self.__n_gen,
            initial_features=self.__initial_features,
            folds_creator=self.__folds_creator,
            objectives=objectives,
            sorting_strategy=self.__sorting_strategy,
            hof_factories=self.__hof_factories,
            mutation=self.__mutation,
            use_clone_repurposing=self.__use_clone_repurposing
        )

    def uses_inner_models(self) -> bool:
        return True

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        res = "NSGA* factory\n"
        res += name_value("Name", self.name()) + "\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += name_value("Population size", self.__pop_size) + "\n"
        res += name_value("Number of features in initial individuals", self.__initial_features) + "\n"
        res += name_value("Number of generations", self.__n_gen) + "\n"
        res += name_value("Crossover probability", self.__mating_prob) + "\n"
        res += name_value("Mutation frequency", self.__mutation_frequency) + "\n"
        res += name_value("Mutation operator", self.__mutation) + "\n"
        res += name_value("Sorting strategy", self.__sorting_strategy) + \
            (" and clone repurposing" if self.__use_clone_repurposing else "") + "\n"
        res += name_value("Hall of fame factories", self.__hof_factories) + "\n"
        res += name_value("Folds creator", self.__folds_creator) + "\n"
        return res
