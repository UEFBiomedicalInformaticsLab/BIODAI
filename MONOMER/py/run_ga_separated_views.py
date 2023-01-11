from collections import Sequence
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_runner.ga_runner import GARunner
from ga_runner.progress_observer import SmartProgressObserver
from input_data.input_data import InputData
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed


# Returned populations are in a list where each element is a population.
def run_ga_separated_views(
        input_data: InputData, ga_runners: list[GARunner],
        folds_creator: InputDataFoldsCreator,
        n_gen: int,
        printer: Printer,
        feature_importance: Sequence[Distribution] = None,
        seed=26542, verbose=True, n_workers=1, initial_view_pops=None,
        workers_printer: Printer = UnbufferedOutPrinter()):
    """Returned populations are in lists where each element is a population."""

    res_pops = []
    res_logbooks = []

    # Fold creator and GA runner use the same seeds for all the views for a more fair comparison.
    folds_creator_seed = seed
    print("folds_creator_seed: " + str(folds_creator_seed))
    ga_runner_seed = None

    view_names = input_data.view_names()

    for i in range(len(view_names)):
        view_name = view_names[i]
        view_input_data = input_data.select_view(view_name)
        folds_list = folds_creator.create_folds_from_input_data(input_data=view_input_data, seed=folds_creator_seed)
        if ga_runner_seed is None:
            ga_runner_seed = random_seed()
            print("ga_runner_seed: " + str(ga_runner_seed))
        if initial_view_pops is None:
            initial_pop = None
        else:
            initial_pop = initial_view_pops[i]
        ga_res = ga_runners[i].run(
            input_data=view_input_data, folds_list=folds_list,
            n_gen=n_gen,
            feature_importance=[feature_importance[i]],
            seed=ga_runner_seed, n_workers=n_workers,
            initial_pop=initial_pop, workers_printer=workers_printer,
            progress_observers=[SmartProgressObserver(printer=printer)])
        new_pop = ga_res.pop
        new_log = ga_res.logbook
        if verbose:
            printer.print("VIEW: " + str(view_name).upper() + "\n" + str(new_log))
        res_pops.append(new_pop)
        res_logbooks.append(new_log)
    return res_pops, res_logbooks
