from collections.abc import Sequence

from deap.tools import Logbook

from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_runner.ga_runner import GARunner
from ga_runner.ga_progress_observer import SmartGAProgressObserver
from hyperparam_manager.view_pops import ViewPops
from input_data.input_data import InputData
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed


# Returned populations are in a list where each element is a population.
def run_ga_separated_views(
        input_data: InputData, ga_runners: Sequence[GARunner],
        folds_creator: InputDataFoldsCreator,
        n_gen: int,
        printer: Printer,
        feature_importance: dict[str,Distribution] = None,
        seed: int = 26542, verbose: bool = True, n_workers: int = 1,
        initial_view_pops: ViewPops = None,
        workers_printer: Printer = UnbufferedOutPrinter()) -> tuple[ViewPops, list[Logbook]]:
    """Returned populations are in lists where each element is a population."""

    res_pops = []
    res_logbooks = []

    # Fold creator and GA runner use the same seeds for all the views for a more fair comparison.
    folds_creator_seed = seed
    printer.print_variable("folds_creator_seed", str(folds_creator_seed))
    ga_runner_seed = None

    view_names = input_data.view_names_seq()

    for i in range(len(view_names)):
        view_name = view_names[i]
        view_input_data = input_data.select_view(view_name)
        folds_list = folds_creator.create_folds_from_input_data(
            input_data=view_input_data, seed=folds_creator_seed, printer=printer)
        if ga_runner_seed is None:
            ga_runner_seed = random_seed()
            printer.print_variable("ga_runner_seed", str(ga_runner_seed))
        if initial_view_pops is None:
            initial_pop = None
        else:
            initial_pop = initial_view_pops.all_individuals_for_view(view_pos=i)
        ga_res = ga_runners[i].run(
            input_data=view_input_data, folds_list=folds_list,
            n_gen=n_gen,
            feature_importance={view_name: feature_importance[view_name]},
            seed=ga_runner_seed, n_workers=n_workers,
            initial_pop=initial_pop, workers_printer=workers_printer,
            progress_observers=[SmartGAProgressObserver(printer=printer)])
        new_pop = ga_res.pop
        new_log = ga_res.logbook
        if verbose:
            printer.print("VIEW: " + str(view_name).upper() + "\n" + str(new_log))
        res_pops.append(new_pop)
        res_logbooks.append(new_log)
    return ViewPops(view_pops=res_pops), res_logbooks
