from collections.abc import Iterable, Sequence

from cross_validation.multi_objective.optimizer.ga_str_utils import nick_paste, pop_nick, gen_nick, \
    sorting_strategy_nick_part, mating_prob_nick, mutation_nick_part, name_paste, pop_name, gen_name, \
    sorting_strategy_name_part, mating_prob_name, mutation_name_part
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import hofs_to_results
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.nsga.nsga_types import NSGA2_TYPE, NSGA3_TYPE
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.flip_ga_runner import FlipGARunner
from ga_runner.progress_observer import SmartProgressObserver
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.individual_with_context import IndividualWithContext
from individual.num_features import NumFeatures
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from univariate_feature_selection.feature_selector_multi_target import DummySelectorMO
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed
from util.utils import name_value


def nsga_optimizer_type(sorting_strategy: SortingStrategy) -> MOOptimizerType:
    if sorting_strategy.basic_algorithm_nick() == NSGA2_TYPE.nick():
        return NSGA2_TYPE
    elif sorting_strategy.basic_algorithm_nick() == NSGA3_TYPE.nick():
        return NSGA3_TYPE
    else:
        raise ValueError("Unknown NSGA type.")


def nsga_nick(
        sorting_strategy: SortingStrategy,
        folds_creator: InputDataFoldsCreator,
        pop_size: int,
        initial_features: NumFeatures,
        n_gen: int,
        use_clone_repurposing: bool,
        mating_prob: float,
        mutation: BitlistMutation,
        mutation_frequency: float
) -> str:
    return nick_paste(parts=[
        nsga_optimizer_type(sorting_strategy=sorting_strategy).nick(),
        folds_creator.nick(),
        pop_nick(pop_size=pop_size),
        initial_features.nick(),
        gen_nick(n_gen=n_gen),
        sorting_strategy_nick_part(sorting_strategy=sorting_strategy,
                                   use_clone_repurposing=use_clone_repurposing),
        mating_prob_nick(mating_prob=mating_prob),
        mutation_nick_part(mutation=mutation, mutation_frequency=mutation_frequency)
        ])


def nsga_name(
        sorting_strategy: SortingStrategy,
        folds_creator: InputDataFoldsCreator,
        pop_size: int,
        initial_features: NumFeatures,
        n_gen: int,
        use_clone_repurposing: bool,
        mating_prob: float,
        mutation: BitlistMutation,
        mutation_frequency: float
) -> str:
    return nsga_optimizer_type(sorting_strategy=sorting_strategy).name() + " (" + name_paste(parts=[
        folds_creator.name(),
        pop_name(pop_size=pop_size),
        initial_features.name(),
        gen_name(n_gen=n_gen),
        sorting_strategy_name_part(sorting_strategy=sorting_strategy,
                                   use_clone_repurposing=use_clone_repurposing),
        mating_prob_name(mating_prob=mating_prob),
        mutation_name_part(mutation=mutation, mutation_frequency=mutation_frequency)
    ]) + ")"


class NsgaStar(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __objectives: Sequence[PersonalObjective]
    __sorting_strategy: SortingStrategy
    __hof_factories: Iterable[HallOfFameFactory]
    __nick: str
    __name: str
    __folds_creator: InputDataFoldsCreator
    __mutation: BitlistMutation
    __initial_features: NumFeatures
    __use_clone_repurposing: bool
    __optimizer_type: MOOptimizerType

    def __init__(self, pop_size, mutation_frequency, mating_prob, n_gen,
                 initial_features: NumFeatures,
                 folds_creator: InputDataFoldsCreator,
                 objectives: Iterable[PersonalObjective],
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
        self.__objectives = list(objectives)
        self.__sorting_strategy = sorting_strategy
        self.__optimizer_type = nsga_optimizer_type(sorting_strategy=sorting_strategy)
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

    def optimize_with_feature_importance(
            self, input_data: InputData, printer: Printer,
            feature_importance: Sequence[Distribution],
            n_proc=1, workers_printer: Printer = UnbufferedOutPrinter(),
            logbook_saver: LogbookSaver = DummyLogbookSaver(),
            feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
            known_solutions: Sequence[IndividualWithContext] = ()) -> Sequence[MultiObjectiveOptimizerResult]:

        # Make sure we do not include outcomes (potentially affecting stratification) that are not in objectives.
        input_data = select_outcomes_in_objectives(input_data=input_data, objectives=self.__objectives)

        collapsed_views = input_data.collapsed_views()

        feature_selector = DummySelectorMO()
        active_features = feature_selector.selection_mask(
            x=collapsed_views, outcomes=input_data.outcomes(), n_proc=n_proc, printer=printer)

        printer.title_print("Creating inner folds")
        folds_list = self.__folds_creator.create_folds_from_input_data(
            input_data=input_data, seed=random_seed(), printer=printer)

        printer.title_print("Running genetic algorithm")
        single_view_runner = FlipGARunner(
            pop_size=self.__pop_size, mutation_frequency=self.__mutation_frequency, mating_prob=self.__mating_prob,
            initial_features=self.__initial_features,
            objectives=self.__objectives,
            sorting_strategy=self.__sorting_strategy,
            active_features=active_features,
            mutation=self.__mutation,
            use_clone_repurposing=self.__use_clone_repurposing
        )
        hofs = [h.create_population_observer() for h in self.__hof_factories]

        if len(known_solutions) > 0:
            initial_pop = [k.modifiable_copy() for k in known_solutions]
        else:
            initial_pop = None

        ga_result = single_view_runner.run(
            input_data=input_data, folds_list=folds_list, return_history=False,
            n_workers=n_proc, workers_printer=workers_printer, n_gen=self.__n_gen,
            feature_importance=feature_importance,
            return_feature_counts=True,
            hofs=hofs,
            progress_observers=[SmartProgressObserver(printer=printer)
                                # , PopNumFeaturesObserver()
                                ],
            initial_pop=initial_pop,
            seed=random_seed()
        )
        printer.print(ga_result.logbook)
        logbook_saver.save(ga_result.logbook)
        feature_names = collapsed_views.columns
        feature_counts_saver.save(feature_counts=ga_result.feature_counts,
                                  feature_names=feature_names)
        return hofs_to_results(hofs)

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

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
        res += name_value("Number of generations", self.__n_gen) + "\n"
        res += name_value("Crossover probability", self.__mating_prob) + "\n"
        res += name_value("Mutation frequency", self.__mutation_frequency) + "\n"
        res += name_value("Mutation operator", self.__mutation) + "\n"
        res += name_value("Sorting strategy", self.__sorting_strategy) + \
            (" and clone repurposing" if self.__use_clone_repurposing else "") + "\n"
        res += name_value("Objectives", self.__objectives) + "\n"
        res += name_value("Hall of fame factories", self.__hof_factories) + "\n"
        res += name_value("Folds creator", self.__folds_creator) + "\n"
        return res
