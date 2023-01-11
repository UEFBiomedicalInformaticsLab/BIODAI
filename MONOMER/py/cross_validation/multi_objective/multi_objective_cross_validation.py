import datetime
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
import socket
import time
from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from consts import DEFAULT_RECURSION_LIMIT, FOLD_RES_DATA_PREFIX, FOLD_RES_DATA_EXTENSION, FEATURE_COUNTS_PREFIX, \
    FEATURE_COUNTS_EXTENSION, FINAL_STR
from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.hof_saver import path_for_saves, save_hof_features, \
    SOLUTION_FEATURES_PREFIX, SOLUTION_FEATURES_EXTENSION, save_hof_fitnesses
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer, \
    MultiObjectiveOptimizerResult, individual_to_line, mo_result_feature_string
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold
from ga_components.feature_counts_saver import FeatureCountsSaver, CsvFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, CsvLogbookSaver
from hall_of_fame.hall_of_fame import HallOfFame
from individual.individual_with_context import IndividualWithContext
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from path_utils import create_optimizer_save_path
from postprocessing.postprocessing import run_postprocessing_archive
from social_space import average_individual
from util.list_math import top_k_positions
from util.printer.printer import CompositePrinter, Printer, NullPrinter, LogPrinter, OutPrinter, UnbufferedOutPrinter
from multi_view_utils import collapse_views
from util.printer.tagged_printer import TaggedPrinter
from util.randoms import set_all_seeds
from util.sequence_utils import transpose
from util.utils import pretty_duration


class FoldSpecificInput(NamedTuple):
    fold_index: int
    save_path: str
    train_data: InputData
    mo_optimizer: MultiObjectiveOptimizer
    fold_processes: int
    seed: int = 68723


FOLD_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT


def print_selected_features_all_results(views, hofs: [MultiObjectiveOptimizerResult], printer: Printer):
    for h in hofs:
        printer.title_print(h.name())
        print_selected_features(views=views, hyperparams=h.hyperparams(), printer=printer)


def print_selected_features_all_hofs(views, hofs: Sequence[HallOfFame], printer: Printer):
    for h in hofs:
        printer.title_print(h.name())
        print_selected_features(views=views, hyperparams=h.hofers(), printer=printer)


def print_selected_features(views, hyperparams: Sequence[IndividualWithContext], printer: Printer,
                            skip_big_lists: bool = True):
    feature_names = collapse_views(views=views).columns
    n_solutions = len(hyperparams)
    printer.print_variable("Number of solutions", n_solutions)
    if n_solutions <= 20 or not skip_big_lists:
        for hp in hyperparams:
            printer.print(individual_to_line(hp=hp, feature_names=feature_names))
    avg_ind = average_individual(individuals=hyperparams)
    a = np.array(avg_ind)
    k = sum(i > 0 for i in a)
    sorted_feature_pos = top_k_positions(a, k)
    printer.print("Features in hall of fame by prevalence:")
    printer.print(mo_result_feature_string(mask=sorted_feature_pos, feature_names=feature_names))


class MultiObjectiveOptimizeOnFolds(ABC):
    """Optimization happens only on training sets. Test sets are ignored.
    Returns a list on folds of tuples (predictors list, hyperparams list, optional hp manager)"""

    @abstractmethod
    def optimize(self, input_data: InputData, folds_list, mo_optimizer: MultiObjectiveOptimizer, save_path, n_proc=1
                 ) -> list[list[MultiObjectiveOptimizerResult]]:
        """External list is for the folds, internal list for the type of hofs."""
        raise NotImplementedError()

    @staticmethod
    def create_fold_printer(save_path: str, fold_index: int) -> Printer:
        str_f = str(fold_index)
        return CompositePrinter([
                TaggedPrinter(tag="Fold " + str_f, inner=UnbufferedOutPrinter()),
                LogPrinter(log_file=save_path + "log_fold_" + str_f + ".txt")])

    @staticmethod
    def create_fold_logbook_saver(save_path: str, fold_index: int) -> LogbookSaver:
        str_f = str(fold_index)
        return CsvLogbookSaver(file=save_path + FOLD_RES_DATA_PREFIX + str_f + "." + FOLD_RES_DATA_EXTENSION)

    @staticmethod
    def create_fold_feat_counts_saver(save_path: str, fold_index: int, save_interval: int = 1) -> FeatureCountsSaver:
        return CsvFeatureCountsSaver(
            file=save_path + FEATURE_COUNTS_PREFIX + str(fold_index) + "." + FEATURE_COUNTS_EXTENSION,
            save_interval=save_interval)

    @staticmethod
    def create_fold_worker_printer(save_path: str, fold_index: int) -> Printer:
        str_f = str(fold_index)
        return CompositePrinter([
            TaggedPrinter(tag="Fold " + str_f, inner=UnbufferedOutPrinter()),
            LogPrinter(log_file=save_path + "workers_log_fold_" + str_f + ".txt")])

    @staticmethod
    def fold_specific_execution(fold_input: FoldSpecificInput) -> [MultiObjectiveOptimizerResult]:
        set_all_seeds(fold_input.seed)
        sys.setrecursionlimit(FOLD_RECURSION_LIMIT)
        f = fold_input.fold_index
        save_path = fold_input.save_path
        fold_printer = MultiObjectiveOptimizeOnFolds.create_fold_printer(save_path=save_path, fold_index=f)
        fold_worker_printer = MultiObjectiveOptimizeOnFolds.create_fold_worker_printer(
            save_path=save_path, fold_index=f)
        fold_logbook_saver = MultiObjectiveOptimizeOnFolds.create_fold_logbook_saver(save_path=save_path, fold_index=f)
        fold_feat_counts_saver = MultiObjectiveOptimizeOnFolds.create_fold_feat_counts_saver(
            save_path=save_path, fold_index=f, save_interval=100)
        fold_printer.title_print("Optimizing on fold number " + str(f))
        fold_workers = fold_input.fold_processes
        fold_printer.print_variable("Number of workers", fold_workers)
        start_time = time.time()
        train_data = fold_input.train_data
        try:
            fold_predictors_with_hyperparams = fold_input.mo_optimizer.optimize(
                input_data=train_data, printer=fold_printer, n_proc=fold_workers,
                workers_printer=fold_worker_printer,
                logbook_saver=fold_logbook_saver,
                feature_counts_saver=fold_feat_counts_saver
            )
            # each fold_predictors_with_hyperparams is a MultiObjectiveOptimizerResult
        except BaseException as e:
            msg = "Exception caught while optimizing on fold " + str(f) + "\n"
            msg += "Original exception: " + str(e) + "\n"
            # msg += automatic_to_string(e) + "\n"
            fold_printer.print(msg)
            raise Exception(msg)
        fold_printer.print("Optimization on fold finished in " + pretty_duration(time.time() - start_time))
        fold_printer.title_print("Feature sets selected")
        print_selected_features_all_results(
            views=fold_input.train_data.views(),
            hofs=fold_predictors_with_hyperparams,
            printer=fold_printer)
        return fold_predictors_with_hyperparams

    @staticmethod
    def create_fold_inputs(input_data: InputData, folds: Folds, mo_optimizer_by_fold: MultiObjectiveOptimizerByFold,
                           save_path, proc_per_fold=1
                           ) -> list[FoldSpecificInput]:
        res = []
        n_folds = folds.n_folds()
        for f in range(n_folds):
            train_data = input_data.select_samples(row_indices=folds.train_indices(fold_number=f))
            fold_input = FoldSpecificInput(
                fold_index=f,
                save_path=save_path,
                mo_optimizer=mo_optimizer_by_fold.optimizer_for_fold(fold_index=f),
                train_data=train_data,
                fold_processes=proc_per_fold)
            res.append(fold_input)
        return res


class MultiObjectiveOptimizeOnFoldsSerial(MultiObjectiveOptimizeOnFolds):

    def optimize(self, input_data: InputData, folds: Folds, mo_optimizer_by_fold: MultiObjectiveOptimizerByFold,
                 save_path, n_proc=1
                 ) -> list[list[MultiObjectiveOptimizerResult]]:
        """Returns a list containing an element for each type of hall of fame, for each type of hall of fame
        there is a different hall of fame for each fold."""
        fold_inputs = self.create_fold_inputs(
            input_data=input_data, folds=folds, mo_optimizer_by_fold=mo_optimizer_by_fold, save_path=save_path,
            proc_per_fold=n_proc)
        non_dominated_predictors_with_hyperparams = []
        recursion_limit = sys.getrecursionlimit()
        for fold_input in fold_inputs:
            non_dominated_predictors_with_hyperparams.append(self.fold_specific_execution(fold_input))
        sys.setrecursionlimit(recursion_limit)
        return transpose(non_dominated_predictors_with_hyperparams)


class MultiObjectiveOptimizeOnFoldsParallel(MultiObjectiveOptimizeOnFolds):

    def optimize(self, input_data: InputData, folds: Folds, mo_optimizer_by_fold: MultiObjectiveOptimizerByFold,
                 save_path, n_proc=1,
                 force_serial=False
                 ) -> list[list[MultiObjectiveOptimizerResult]]:
        n_folds = folds.n_folds()
        proc_per_fold = n_proc // n_folds
        if force_serial or n_folds == 1 or proc_per_fold < 1:
            # Falls back to serial if there is just one fold or there is less than one process per fold.
            return MultiObjectiveOptimizeOnFoldsSerial().optimize(
                input_data=input_data, folds=folds, mo_optimizer_by_fold=mo_optimizer_by_fold,
                save_path=save_path, n_proc=n_proc)
        else:
            fold_inputs = self.create_fold_inputs(
                input_data=input_data, folds=folds, mo_optimizer_by_fold=mo_optimizer_by_fold, save_path=save_path,
                proc_per_fold=proc_per_fold)
            cpu_count = multiprocessing.cpu_count()
            n_workers = min(cpu_count, n_folds)
            # Not greater than n_proc otherwise the serial version would have been used.
            with ProcessPoolExecutor(max_workers=n_workers) as workers_pool:
                non_dominated_predictors_with_hyperparams = workers_pool.map(
                    self.fold_specific_execution, fold_inputs, chunksize=1)
            try:
                res = []
                for i in non_dominated_predictors_with_hyperparams:
                    res.append(i)
                # Not using list constructor since we get a RecursionError while creating a list
                # from: <generator object _chain_from_iterable_of_lists> while collecting results of classic MO GA.
                # res = list(non_dominated_predictors_with_hyperparams)
            except RecursionError as e:
                raise RecursionError("Original error: " + str(e) + "\n" +
                                     "while appending:\n" +
                                     str(i))
            res = transpose(res)
            return res


class MOEvaluationRes:
    __evaluation_res: list
    __final_optimization: [MultiObjectiveOptimizerResult]

    def __init__(self, evaluation_res: list, final_optimization: [MultiObjectiveOptimizerResult]):
        self.__evaluation_res = evaluation_res
        self.__final_optimization = final_optimization

    def evaluation_res(self) -> list:
        return self.__evaluation_res

    def final_optimization(self) -> [MultiObjectiveOptimizerResult]:
        return self.__final_optimization

    def __str__(self):
        return "Evaluation result: " + str(self.__evaluation_res) + "\n" +\
               "Final optimization:\n" + str(self.__final_optimization) + "\n"


def create_final_optimizer_printer(optimizer_save_path: str, additional_printer: Printer = NullPrinter()) -> Printer:
    log_printer = CompositePrinter([LogPrinter(log_file=optimizer_save_path + "log_final.txt"), additional_printer])
    return CompositePrinter([OutPrinter(), log_printer])


def do_final_optimization(
        input_data: InputData,
        mo_optimizer: MultiObjectiveOptimizer,
        objectives: [PersonalObjective],
        save_path: str,
        additional_printer: Printer = NullPrinter(),
        n_proc=1) -> [MultiObjectiveOptimizerResult]:
    optimizer_nick = mo_optimizer.nick()
    optimizer_save_path = create_optimizer_save_path(save_path=save_path, optimizer_nick=optimizer_nick)
    printer = create_final_optimizer_printer(
        optimizer_save_path=optimizer_save_path, additional_printer=additional_printer)
    printer.title_print("Optimizing final model on whole dataset")
    start_time = time.time()
    workers_log_printer = LogPrinter(log_file=optimizer_save_path + "workers_log.txt")
    workers_printer = CompositePrinter([OutPrinter(), workers_log_printer])
    final_optimization = mo_optimizer.optimize(
        input_data=input_data, printer=printer, n_proc=n_proc, workers_printer=workers_printer)
    printer.print_variable(
        mo_optimizer.name() + " final optimization execution time", pretty_duration(time.time() - start_time))
    printer.title_print("Feature sets selected on whole dataset")
    print_selected_features_all_results(
        views=input_data.views(),
        hofs=final_optimization,
        printer=printer)
    printer.title_print("Saving features and fitnesses for each selected individual to files.")
    for hof in final_optimization:
        path_saves = path_for_saves(base_path=save_path, optimizer_nick=optimizer_nick, hof_nick=hof.nick())
        save_hof_features(path_saves=path_saves,
                          file_name=SOLUTION_FEATURES_PREFIX + FINAL_STR + SOLUTION_FEATURES_EXTENSION,
                          feature_names=input_data.collapsed_feature_names(),
                          hof=hof)
        save_hof_fitnesses(path_saves=path_saves,
                           file_name="solution_fitnesses_" + FINAL_STR + ".csv",
                           hof=hof,
                           objectives=objectives,
                           x_train=input_data.views(),
                           y_train=input_data.outcomes_data_dict())
    return final_optimization


def optimize_with_evaluation(input_data: InputData,
                             folds: Folds,
                             mo_optimizer_by_fold: MultiObjectiveOptimizerByFold,
                             mo_evaluators: Iterable[MultiObjectiveCrossEvaluator],
                             save_path: str,
                             additional_printer: Printer = NullPrinter(),
                             run_folds_in_parallel=True,
                             n_proc=1
                             ) -> list:
    optimizer_nick = mo_optimizer_by_fold.nick()
    optimizer_save_path = create_optimizer_save_path(save_path=save_path, optimizer_nick=optimizer_nick)
    log_printer = CompositePrinter([LogPrinter(log_file=optimizer_save_path + "log.txt"), additional_printer])
    printer = CompositePrinter([OutPrinter(), log_printer])
    printer.title_print("Running " + mo_optimizer_by_fold.name())
    log_printer.print_variable("Date", datetime.date.today())
    printer.print_variable("Machine", socket.gethostname())
    log_printer.print_variable("CPUs detected in the system", multiprocessing.cpu_count())
    log_printer.print_variable("Number of outer folds", folds.n_folds())
    printer.title_print("Optimizer details")
    printer.print(mo_optimizer_by_fold)
    start_time = time.time()
    printer.title_print("Starting multi-objective optimization on different folds")
    if run_folds_in_parallel:
        folds_optimizer = MultiObjectiveOptimizeOnFoldsParallel()
    else:
        folds_optimizer = MultiObjectiveOptimizeOnFoldsSerial()
    folds_hofs = folds_optimizer.optimize(
        input_data=input_data, folds=folds, mo_optimizer_by_fold=mo_optimizer_by_fold, save_path=optimizer_save_path,
        n_proc=n_proc)
    printer.print("Computation on different folds finished in " + pretty_duration(time.time() - start_time))
    evaluation_res = []
    printer.title_print("Applying evaluators")
    for hof in folds_hofs:
        printer.title_print("Evaluating hall of fame " + hof[0].name())
        for evaluator in mo_evaluators:
            printer.title_print("Applying " + evaluator.name())
            evaluation = evaluator.evaluate(
                input_data=input_data, folds=folds,
                non_dominated_predictors_with_hyperparams=hof,
                optimizer_nick=mo_optimizer_by_fold.nick(), printer=printer)
            evaluation_res.append(evaluation)

    run_postprocessing_archive(optimizer_dir=optimizer_save_path, printer=printer)
    return evaluation_res
