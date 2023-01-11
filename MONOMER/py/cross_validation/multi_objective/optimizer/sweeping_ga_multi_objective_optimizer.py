from collections import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.ga_str_utils import nick_paste, pop_nick, \
    sorting_strategy_nick_part, mating_prob_nick, mutation_nick_part, name_paste, pop_name, \
    sorting_strategy_name_part, mating_prob_name, mutation_name_part
from cross_validation.multi_objective.optimizer.mo_optimizer_type import ConcreteMOOptimizerType, MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult, \
    hofers_to_results
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.sweeping_strategy import SweepingStrategy
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.feature_counts_saver import DummyFeatureCountsSaver, FeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from ga_components.sorter.sorting_strategy import SortingStrategy
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.num_features import NumFeatures
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from run_ga_2_steps import run_ga_2_steps
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.utils import name_value


class SweepingGAMultiObjectiveOptimizer(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __objectives: Iterable[PersonalObjective]
    __sorting_strategy: SortingStrategy
    __nick: str
    __name: str
    __sweeping_strategy: SweepingStrategy
    __hof_factories: Iterable[HallOfFameFactory]
    __folds_creator: InputDataFoldsCreator
    __mutation: BitlistMutation
    __initial_features: NumFeatures
    __use_clone_repurposing: bool

    __optimizer_type = ConcreteMOOptimizerType(
        uses_inner_models=True, nick="Sweeping", name="Sweeping genetic algorithm multi-view multi-objective optimizer")

    def __init__(self, pop_size, mutation_frequency, mating_prob,
                 initial_features: NumFeatures,
                 folds_creator: InputDataFoldsCreator,
                 objectives: Iterable[PersonalObjective],
                 sweeping_strategy: SweepingStrategy,
                 sorting_strategy: SortingStrategy,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 mutation: BitlistMutation = FlipMutation(),
                 use_clone_repurposing: bool = False):
        self.__pop_size = pop_size
        self.__mutation_frequency = mutation_frequency
        self.__mating_prob = mating_prob
        self.__initial_features = initial_features
        self.__folds_creator = folds_creator
        self.__sweeping_strategy = sweeping_strategy
        self.__objectives = objectives
        self.__sorting_strategy = sorting_strategy
        self.__hof_factories = hof_factories
        self.__mutation = mutation
        self.__use_clone_repurposing = use_clone_repurposing
        self.__nick = nick_paste(parts=[self.__optimizer_type.nick(),
                                        self.__folds_creator.nick(),
                                        pop_nick(pop_size=pop_size),
                                        initial_features.nick(),
                                        "gen" + sweeping_strategy.nick(),
                                        sorting_strategy_nick_part(sorting_strategy=sorting_strategy,
                                                                   use_clone_repurposing=use_clone_repurposing),
                                        mating_prob_nick(mating_prob=mating_prob),
                                        mutation_nick_part(mutation=mutation, mutation_frequency=mutation_frequency)
                                        ])
        self.__name = self.__optimizer_type.name() + " (" + name_paste(parts=[
            self.__folds_creator.name(),
            pop_name(pop_size=pop_size),
            initial_features.name(),
            "gen " + sweeping_strategy.name(),
            sorting_strategy_name_part(sorting_strategy=sorting_strategy,
                                       use_clone_repurposing=use_clone_repurposing),
            mating_prob_name(mating_prob=mating_prob),
            mutation_name_part(mutation=mutation, mutation_frequency=mutation_frequency)
        ]) + ")"

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: Sequence[Distribution],
                                         n_proc=1, workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                                         ) -> Sequence[MultiObjectiveOptimizerResult]:
        ga_result = run_ga_2_steps(
            input_data=input_data, pop_size=self.__pop_size,
            mutating_prob=self.__mutation_frequency, mating_prob=self.__mating_prob,
            sweeping_strategy=self.__sweeping_strategy,
            initial_features=self.__initial_features,
            folds_creator=self.__folds_creator,
            n_workers=n_proc,
            return_history=False,
            objectives=self.__objectives,
            sorting_strategy=self.__sorting_strategy,
            feature_importance=feature_importance,
            printer=printer, workers_printer=workers_printer,
            hof_factories=self.__hof_factories
        )
        logbook_saver.save(ga_result.logbook)
        return hofers_to_results(ga_result.hofers)

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick

    def __str__(self) -> str:
        res = ""
        res += name_value("Name", self.name()) + "\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += name_value("Population size", self.__pop_size) + "\n"
        res += name_value("Number of features in initial individuals", self.__initial_features) + "\n"
        res += name_value("Number of generations per sweep", self.__sweeping_strategy) + "\n"
        res += name_value("Crossover probability", self.__mating_prob) + "\n"
        res += name_value("Mutation frequency", self.__mutation_frequency) + "\n"
        res += name_value("Mutation operator", self.__mutation) + "\n"
        res += name_value("Sorting strategy", self.__sorting_strategy) + \
                         (" and clone repurposing" if self.__use_clone_repurposing else "") + "\n"
        res += name_value("Objectives", self.__objectives) + "\n"
        res += name_value("Hall of fame factories", self.__hof_factories) + "\n"
        res += name_value("Folds creator", self.__folds_creator) + "\n"
        return res

    def optimizer_type(self) -> MOOptimizerType:
        return SweepingGAMultiObjectiveOptimizer.__optimizer_type
