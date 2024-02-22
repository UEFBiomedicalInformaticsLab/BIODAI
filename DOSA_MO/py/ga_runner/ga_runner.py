from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import NamedTuple, List

import numpy
from deap import base
from deap import tools
from deap.tools import History, Logbook

from evaluator.individual_updater import IndividualUpdater
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.explored_features import ExploredFeatures
from ga_runner.progress_observer import ProgressObserver
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from hyperparam_manager.hyperparam_manager import HyperparamManager
from ga_components.var_and import varAnd
from individual.peculiar_individual import PeculiarIndividual
from ga_strategy.ga_strategy import GAStrategy
from individual.peculiar_individual_with_context import contextualize_all
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from social_space import sum_of_individuals
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import set_all_seeds
from util.sparse_bool_list_by_set import union


class GAResult(NamedTuple):
    pop: List[PeculiarIndividual]  # TODO We can consider removing this to reduce memory usage (can be optional hof).
    hp_manager: HyperparamManager
    logbook: Logbook
    hofs: Sequence[Hofers]  # TODO Possibly not needed since hofs are passed from outside.
    history: History
    feature_counts: Sequence[Sequence[int]]


class GARunner(ABC):
    __objectives: Iterable[PersonalObjective]
    __sorting_strategy: SortingStrategy
    __use_clone_repurposing: bool

    def __init__(self, pop_size, mating_prob, mutation_frequency,
                 objectives: Iterable[PersonalObjective],
                 sorting_strategy: SortingStrategy,
                 use_clone_repurposing: bool = False):
        self.__pop_size = pop_size
        self.__mating_prob = mating_prob
        self.__mutation_frequency = mutation_frequency
        self.__objectives = objectives
        self.__sorting_strategy = sorting_strategy
        self.__use_clone_repurposing = use_clone_repurposing

    def run(self, input_data: InputData, folds_list, n_gen: int,
            feature_importance: Sequence[Distribution] = None,
            seed=2547, n_workers=1,
            return_history=False,
            return_feature_counts=False,
            initial_pop=None,
            workers_printer: Printer = UnbufferedOutPrinter(),
            hofs: Iterable[HallOfFame] = (),
            progress_observers: [ProgressObserver] = ()) -> GAResult:
        """We pass hofs instead of hofs factories so that a hof can be used for more than one run."""
        ga_strategy = self._create_ga_strategy(
            input_data=input_data, folds_list=folds_list, n_workers=n_workers,
            feature_importance=feature_importance,
            workers_printer=workers_printer)
        return run_ga(
            ga_strategy=ga_strategy, pop_size=self.pop_size(), n_gen=n_gen,
            seed=seed, return_history=return_history, initial_pop=initial_pop,
            return_feature_counts=return_feature_counts,
            hofs=hofs, progress_observers=progress_observers)

    def pop_size(self):
        return self.__pop_size

    def mating_prob(self):
        return self.__mating_prob

    def mutation_frequency(self):
        return self.__mutation_frequency

    def _objectives(self):
        return self.__objectives

    def use_clone_repurposing(self) -> bool:
        return self.__use_clone_repurposing

    @abstractmethod
    def _create_ga_strategy(self, input_data: InputData, folds_list, n_workers,
                            feature_importance: Sequence[Distribution],
                            workers_printer: Printer = UnbufferedOutPrinter()):
        raise NotImplementedError()

    def sorting_strategy(self) -> SortingStrategy:
        return self.__sorting_strategy


def create_toolbox(ga_strategy: GAStrategy):
    toolbox = base.Toolbox()
    toolbox.register("population", tools.initRepeat, list, ga_strategy.create_individual)
    toolbox.register("mate", ga_strategy.mate)
    toolbox.register("mutate", ga_strategy.mutate)
    return toolbox


def check_invalids(pop):
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    return len(invalid_ind) > 0


def calc_n_features(views):
    res = 0
    for v in views:
        res += len(views[v].columns)
    return res


def create_tools_statistics_from_fitness(index):
    return tools.Statistics(key=lambda ind: ind.fitness.values[index])


def pop_short_print(pop):
    for i in pop:
        print(i.fitness)


# If initial pop is passed pop size is ignored.
def run_ga(ga_strategy: GAStrategy, pop_size: int, n_gen: int, seed=2547, return_history=False,
           initial_pop: Sequence[PeculiarIndividual]=None, return_feature_counts=False, hofs: Iterable[HallOfFame] = (),
           progress_observers: [ProgressObserver] = ()) -> GAResult:
    """Explored features are computed before the select.
    Feature counts are computed after the select."""

    set_all_seeds(seed)

    toolbox = create_toolbox(ga_strategy)

    objectives = ga_strategy.objectives()
    stat_names = []

    stat_tools = {}
    o_index = 0
    for o in objectives:
        stat_tools[o.name()] = create_tools_statistics_from_fitness(o_index)
        o_index += 1
        stat_names.append(o.name())

    # Adding statistics
    for_stats = ga_strategy.to_be_added_to_stats()
    for a in for_stats:
        stat_tools[a] = tools.Statistics(key=for_stats[a])
        stat_names.append(a)

    stats = tools.MultiStatistics(**stat_tools)

    stats.register("min", numpy.min, axis=0)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("std", numpy.std, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "gen_feats", "explored"]
    # gen_feats are the features of newly evaluated individuals, not of the whole population.
    logbook.header.extend(stat_names)

    for on in stat_names:
        logbook.chapters[on].header = "min", "avg", "max", "std"

    pop = []
    if initial_pop is not None:
        pop = initial_pop
        initial_pop = None  # Help gc
        if len(pop) > pop_size:
            pop = pop[:pop_size]
    if len(pop) < pop_size:
        pop.extend(toolbox.population(n=pop_size-len(pop)))

    evaluator = ga_strategy.evaluator()
    hp_manager = evaluator.hp_manager()
    # cross_evaluator.add_stat_creator(StatsFromConfusion())

    objectives = ga_strategy.objectives()

    individual_updater = IndividualUpdater(evaluator=evaluator, objectives=objectives)
    invalid_ind = individual_updater.eval_invalid(pop=pop)  # Evaluate the individuals with an invalid fitness

    pop = ga_strategy.sorting_strategy_before_selection(individuals=pop)
    # We assign the secondary attributes that will be used by the first tournament.
    # We use the before selection strategy that is guaranteed to assign the attributes.

    contextualized = contextualize_all(pop, hp_manager)
    for h in hofs:
        h.update(contextualized)

    if return_history:
        history = History()
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)
        history.update(pop)
    else:
        history = None

    feat_count = sum_of_individuals(individuals=pop, hp_manager=hp_manager)
    # Compile statistics about the population
    gen_feats_list = [f > 0 for f in feat_count]
    explored = ExploredFeatures()
    explored.update([gen_feats_list])
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), gen_feats=sum(gen_feats_list),
                   explored=explored.num_explored_features(), **record)
    if return_feature_counts:
        feature_counts = [feat_count]
    else:
        feature_counts = None

    for o in progress_observers:
        o.notify_initial_pop(pop)

    # Begin the generational process
    for gen in range(1, n_gen):

        offspring = ga_strategy.tournament(pop=pop, k=len(pop))

        for o in progress_observers:
            o.notify_tournament_offsprings(offspring)

        offspring = varAnd(
            population=offspring, toolbox=toolbox, cxpb=ga_strategy.mating_prob(), mutpb=1.0, crossover_first=False)

        for o in progress_observers:
            o.notify_modified_offsprings(offspring)

        pop = pop + offspring

        pop = ga_strategy.clone_repurposing(pop=pop)

        invalid_ind = individual_updater.eval_invalid(pop=pop)  # Evaluate the individuals with an invalid fitness
        contextualized = contextualize_all(invalid_ind, hp_manager)
        gen_feats_list = union(contextualized)
        explored.update([gen_feats_list])
        for h in hofs:
            h.update(contextualized)

        pop = ga_strategy.sorting_strategy_before_selection(individuals=pop)

        for o in progress_observers:
            o.notify_pop_before_select(pop)

        # Select the next generation population from parents and offspring
        pop = ga_strategy.select(pop=pop, pop_size=pop_size)

        pop = ga_strategy.sorting_strategy_after_selection(individuals=pop)

        for o in progress_observers:
            o.notify_pop_after_select(pop)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), gen_feats=sum(gen_feats_list),
                       explored=explored.num_explored_features(),  **record)
        # We should be able to use strings for column names with the form **{attr_name: attr_value}

        if return_feature_counts:
            feature_counts.append(sum_of_individuals(individuals=pop, hp_manager=hp_manager))

        for po in progress_observers:
            po.notify_generation_end(gen=gen)

    evaluator.cleanup()

    contextualized_final_pop = contextualize_all(pop, hp_manager)
    for h in hofs:
        h.signal_final(final_elems=contextualized_final_pop)

    res_hofs = [h.hofers() for h in hofs]

    return GAResult(pop=pop, hp_manager=hp_manager, logbook=logbook, history=history,
                    feature_counts=feature_counts, hofs=res_hofs)
