from collections.abc import Iterable
from typing import NamedTuple, Sequence

from deap.tools import Logbook, History
from joblib.numpy_pickle_utils import xrange

from cross_validation.multi_objective.multi_objective_cross_validation import print_selected_features_all_hofs
from cross_validation.multi_objective.optimizer.sweeping_strategy import SweepingStrategy
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.progress_observer import SmartProgressObserver
from hall_of_fame.hofers import Hofers
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.num_features import NumFeatures
from individual.individual_with_context import IndividualWithContext
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from ga_runner.flip_ga_runner import FlipGARunner
from ga_runner.select_ga_runner import SelectGARunner
from hyperparam_manager.select_hp_manager import SelectHpManager
from individual.peculiar_individual_with_context import contextualize_all
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from univariate_feature_selection.univariate_feature_selection import compute_active_features_mv_multi_target
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed, set_all_seeds
from run_ga_separated_views import run_ga_separated_views
import deprecation


def view_pops_from_resampling(master_pop, view_pops):
    n_views = len(view_pops)
    res_pops = [[] for _ in xrange(n_views)]
    for m_ind in master_pop:
        for view_i in range(n_views):
            m_ind_pointer = m_ind[view_i]
            res_pops[view_i].append(view_pops[view_i][m_ind_pointer])
    return res_pops


# Creates a conversion of the master individual to a mask individual with a bit for each active feature in
# each view.
@deprecation.deprecated(details="Deprecated since not used and tested anymore.")
def master_individual_to_mask(master_individual: PeculiarIndividualByListlike, view_pops):
    res = PeculiarIndividualByListlike(n_objectives=master_individual.n_objectives())
    n_views = len(view_pops)
    for view_i in range(n_views):
        m_ind_pointer = master_individual[view_i]
        res.append(view_pops[view_i][m_ind_pointer])
    res.fitness = master_individual.fitness
    res.set_stats(master_individual.get_stats())
    res.set_predictors(master_individual.get_predictors())
    return res


class RunGA2StepsResult(NamedTuple):
    pop: list[IndividualWithContext]
    logbook: Logbook
    hofers: Sequence[Hofers]
    history: History


def run_ga_2_steps(
        input_data: InputData, pop_size, mutating_prob, mating_prob,
        sweeping_strategy: SweepingStrategy,
        initial_features: NumFeatures,
        folds_creator: InputDataFoldsCreator,
        objectives: Iterable[PersonalObjective],
        sorting_strategy: SortingStrategy,
        printer: Printer,
        feature_importance: Sequence[Distribution] = None,
        seed=844, n_workers=1,
        hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
        return_history=False,
        workers_printer: Printer = UnbufferedOutPrinter(),
        bitlist_mutation: BitlistMutation = FlipMutation(),
        use_clone_repurposing: bool = False) -> RunGA2StepsResult:

    printer.title_print("Starting 2-steps genetic algorithm")

    set_all_seeds(seed)

    # Make sure we do not include outcomes (potentially affecting feature selection) not in objectives.
    input_data = select_outcomes_in_objectives(input_data=input_data, objectives=objectives)

    printer.title_print("Computing local active features")
    active_features_by_view = compute_active_features_mv_multi_target(
        views=input_data.views(), y=input_data.collapsed_outcomes(), printer=printer, n_proc=n_workers)

    # check_masks(views, active_features_by_view)  # TODO Debug code, remove when stable.

    printer.title_print("Creating inner folds")
    folds_list = folds_creator.create_folds_from_input_data(input_data=input_data, seed=random_seed())

    single_view_runners = []
    for i in range(input_data.n_views()):
        single_view_runner = FlipGARunner(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutating_prob,
            initial_features=initial_features,
            objectives=objectives,
            sorting_strategy=sorting_strategy,
            active_features=active_features_by_view[i],
            mutation=bitlist_mutation,
            use_clone_repurposing=use_clone_repurposing
        )
        single_view_runners.append(single_view_runner)

    view_pops = None
    master_result = None
    result_hofs = [h.create_population_observer() for h in hof_factories]

    for i in range(sweeping_strategy.num_sweeps()):

        n_gen = sweeping_strategy.generations(i)

        master_result = None  # Help gc

        printer.title_print("Starting sweep number " + str(i+1))

        view_pops, _ = run_ga_separated_views(
            input_data=input_data, ga_runners=single_view_runners, folds_creator=folds_creator,
            n_gen=n_gen, feature_importance=feature_importance,
            seed=random_seed(), n_workers=n_workers, initial_view_pops=view_pops, printer=printer,
            workers_printer=workers_printer)

        select_hp_manager = SelectHpManager(view_pops=view_pops)

        master_runner = SelectGARunner(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutating_prob,
            hp_manager=select_hp_manager, objectives=objectives,
            sorting_strategy=sorting_strategy, use_clone_repurposing=use_clone_repurposing)

        master_result = master_runner.run(
            input_data=input_data, folds_list=folds_list, n_gen=n_gen, seed=random_seed(), n_workers=n_workers,
            hofs=result_hofs,
            return_history=return_history, workers_printer=workers_printer,
            progress_observers=[SmartProgressObserver(printer=printer)])

        printer.print("MASTER\n" + str(master_result.logbook))

        printer.title_print("Current halls of fame")
        print_selected_features_all_hofs(
            views=input_data.views(),
            hofs=result_hofs,
            printer=printer)

        view_pops = view_pops_from_resampling(master_pop=master_result.pop, view_pops=view_pops)
        # TODO: perhaps we can use the HOF instead of the final population.

    res_pop = contextualize_all(hps=master_result.pop, hp_manager=master_result.hp_manager)

    hofers = [h.hofers() for h in result_hofs]

    return RunGA2StepsResult(
        pop=res_pop, logbook=master_result.logbook, hofers=hofers, history=master_result.history)
