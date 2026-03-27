from collections.abc import Iterable
from typing import NamedTuple, Sequence

from deap.tools import Logbook, History

from cross_validation.multi_objective.multi_objective_cross_validation import print_selected_features_all_hofs
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.ga_progress_observer import SmartGAProgressObserver
from hall_of_fame.hofers import Hofers
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.num_features import NumFeatures
from individual.individual_with_context import IndividualWithContext
from ga_runner.flip_ga_runner import FlipGARunner
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from objective.social_objective import PersonalObjective
from run_ga.master_runner import ResamplingMaster, MasterRunner
from run_ga.run_ga_separated_views import run_ga_separated_views
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed, set_all_seeds


class RunSweepsResult(NamedTuple):
    pop: Sequence[IndividualWithContext]
    logbook: Logbook
    hofers: Sequence[Hofers]
    history: History


def run_sweeps(
        input_data: InputData,
        pop_size: int,
        mutating_prob: float,
        mating_prob: float,
        sweeping_strategy: GenerationsStrategy,
        initial_features: NumFeatures,
        folds_creator: InputDataFoldsCreator,
        objectives: Iterable[PersonalObjective],
        sorting_strategy: SortingStrategy,
        printer: Printer,
        feature_importance: dict[str,Distribution] = None,
        seed=844,
        n_workers: int = 1,
        hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
        return_history: bool = False,
        workers_printer: Printer = UnbufferedOutPrinter(),
        bitlist_mutation: BitlistMutation = FlipMutation(),
        use_clone_repurposing: bool = False,
        master_runner: MasterRunner = ResamplingMaster()) -> RunSweepsResult:

    printer.title_print("Starting 2-steps genetic algorithm")

    set_all_seeds(seed)

    # Make sure we do not include outcomes (potentially affecting feature selection) not in objectives.
    input_data = select_outcomes_in_objectives(input_data=input_data, objectives=objectives)

    printer.title_print("Creating inner folds")
    folds_list = folds_creator.create_folds_from_input_data(input_data=input_data, seed=random_seed(), printer=printer)

    single_view_runners = []
    for i in range(input_data.n_views()):
        single_view_runner = FlipGARunner(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutating_prob,
            initial_features=initial_features,
            objectives=objectives,
            sorting_strategy=sorting_strategy,
            mutation=bitlist_mutation,
            use_clone_repurposing=use_clone_repurposing
        )
        single_view_runners.append(single_view_runner)

    view_pops = None
    master_result = None
    result_hofs = [h.create_population_observer() for h in hof_factories]

    for i in range(sweeping_strategy.num_sweeps()):

        n_gen = sweeping_strategy.sweep_generations(i)

        master_result = None  # Help gc

        printer.title_print("Starting sweep number " + str(i+1))

        view_pops, _ = run_ga_separated_views(
            input_data=input_data, ga_runners=single_view_runners, folds_creator=folds_creator,
            n_gen=n_gen, feature_importance=feature_importance,
            seed=random_seed(), n_workers=n_workers, initial_view_pops=view_pops, printer=printer,
            workers_printer=workers_printer)

        printer.title_print("Starting master runner " + master_runner.name())
        master_result, view_pops = master_runner.run_master(
            input_data=input_data,
            view_pops=view_pops,
            pop_size=pop_size,
            mutating_prob=mutating_prob,
            mating_prob=mating_prob,
            objectives=objectives,
            sorting_strategy=sorting_strategy,
            folds_list=folds_list,
            n_gen=n_gen,
            result_hofs=result_hofs,
            printer=printer,
            bitlist_mutation=bitlist_mutation,
            initial_features=initial_features,
            use_clone_repurposing=use_clone_repurposing,
            workers_printer=workers_printer,
            return_history=return_history,
            n_workers=n_workers)

        printer.print("MASTER\n" + str(master_result.logbook))

        printer.title_print("Current halls of fame")
        print_selected_features_all_hofs(
            views=input_data.views(),
            hofs=result_hofs,
            printer=printer)

    final_result = master_result
    view_pops = None  # Help GC.
    master_result = None  # Help GC.
    res_pop = final_result.hp_manager.contextualize_all(pop=final_result.pop)

    if sweeping_strategy.concatenated_generations() > 0:
        printer.title_print("Starting concatenated optimization")

        initial_pop = [i.modifiable_copy() for i in res_pop]
        res_pop = None  # Help GC

        concatenated_runner = FlipGARunner(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutating_prob,
            initial_features=initial_features,
            objectives=objectives,
            sorting_strategy=sorting_strategy,
            mutation=bitlist_mutation,
            use_clone_repurposing=use_clone_repurposing
        )
        final_result = concatenated_runner.run(
            input_data=input_data, folds_list=folds_list,
            n_gen=sweeping_strategy.concatenated_generations(),
            seed=random_seed(), n_workers=n_workers,
            initial_pop=initial_pop, workers_printer=workers_printer,
            progress_observers=[SmartGAProgressObserver(printer=printer)],
            hofs=result_hofs)
        printer.print("CONCATENATED\n" + str(final_result.logbook))

    res_pop = final_result.hp_manager.contextualize_all(pop=final_result.pop)

    hofers = [h.hofers() for h in result_hofs]

    return RunSweepsResult(
        pop=res_pop, logbook=final_result.logbook, hofers=hofers, history=final_result.history)
