import os
import shutil
from collections.abc import Sequence

from cross_validation.folds import Folds, load_folds
from cross_validation.multi_objective.cross_evaluator.folds_saver import FOLDS_FILE_NAME
from location_manager.location_consts import HOFS_STR
from folds_creator.default_folds_creator import default_folds_creator
from hall_of_fame.fronts import PARETO_NICK
from input_data.input_creators_archive import INPUT_CREATORS_DICT
from input_data.input_data import InputData
from load_omics_views import MRNA_NAME
from setup.allowed_names import STRATIFIED_K_FOLD_NAME, LOAD_FOLD_NAME, AUTO_FOLD_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG, OUTER_N_FOLDS_SMALL
from util.printer.printer import Printer, OutPrinter, NullPrinter


def outer_folds_num(use_big_setup: bool) -> int:
    if use_big_setup:
        return OUTER_N_FOLDS_BIG
    else:
        return OUTER_N_FOLDS_SMALL


def create_outer_folds(outer_n_folds: int, input_data: InputData, n_repeats: int = 1,
                       printer: Printer = NullPrinter(), seed: int = 365):
    return Folds(test_sets=[
        f[1] for f in create_outer_folds_list(
            outer_n_folds=outer_n_folds, input_data=input_data, n_repeats=n_repeats, printer=printer, seed=seed)])


def create_outer_folds_list(outer_n_folds: int, input_data: InputData, n_repeats: int = 1,
                            printer: Printer = NullPrinter(),
                            seed: int = 365):
    outer_folds_creator = default_folds_creator(n_folds=outer_n_folds, n_repeats=n_repeats)
    return outer_folds_creator.create_folds_from_input_data(input_data=input_data, printer=printer, seed=seed)


def combine_objective_strings(objective_strings: Sequence[str]) -> str:
    temp_strings = list(objective_strings)
    temp_strings.sort()
    objectives_str = ""
    for o in temp_strings:
        if objectives_str != "":
            objectives_str += "_"
        objectives_str += o
    return objectives_str


def load_input_data(dataset_name: str, views_to_use: [str] = (MRNA_NAME, ),
                    printer: Printer = OutPrinter()) -> InputData:
    if dataset_name in INPUT_CREATORS_DICT:
        input_creator = INPUT_CREATORS_DICT[dataset_name]
    else:
        raise ValueError("Unknown dataset " + str(dataset_name))
    printer.title_print("Loading " + input_creator.nick() + " cohort")
    res = input_creator.create(views_to_load=views_to_use, printer=printer)
    printer.title_print("Loaded " + str(res.n_samples()) + " samples.")
    return res


def save_config_file(config_file: str, destination_path: str, printer: Printer):
    if config_file is not None:
        printer.print("Copying config file to run directory")
        printer.print_variable("Copy source", config_file)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(config_file, destination_path + "config.ini")


def setup_to_folds(setup: EvaluationSetup, input_data: InputData, printer: Printer, seed: int = 365) -> Folds:
    base_dir = setup.load_base_dir()
    outer_folds_str = setup.outer_folds()
    if isinstance(base_dir, str):
        if outer_folds_str == STRATIFIED_K_FOLD_NAME:
            raise ValueError("Existing base directory but new stratification requested.")
        elif outer_folds_str == LOAD_FOLD_NAME or outer_folds_str == AUTO_FOLD_NAME:
            base_dir = setup.load_base_dir()
            file_to_load = os.path.join(base_dir, FOLDS_FILE_NAME)
            printer.print("Loading existing folds from " + str(file_to_load))
            return load_folds(file_path=file_to_load)
        else:
            raise ValueError("Unexpected outer folds setup.")
    else:
        if outer_folds_str == STRATIFIED_K_FOLD_NAME or outer_folds_str == AUTO_FOLD_NAME:
            if setup.outer_n_folds() is None:
                outer_n_folds = outer_folds_num(use_big_setup=setup.use_big_defaults())
            else:
                outer_n_folds = setup.outer_n_folds()
            printer.print("Creating " + str(outer_n_folds) + " stratified folds.")
            outer_folds = create_outer_folds(
                outer_n_folds=outer_n_folds, input_data=input_data,
                n_repeats=setup.cv_repeats(), printer=printer, seed=seed)
            return outer_folds
        elif outer_folds_str == LOAD_FOLD_NAME:
            raise ValueError("Load of folds requested but no valid base directory received.")
        else:
            raise ValueError("Unexpected outer folds setup.")


def choose_hof_nick(previous_optimizer_dir: str, hofs_priority: [str] = (PARETO_NICK,)) -> str:
    if os.path.isdir(previous_optimizer_dir):
        for hof_n in hofs_priority:
            hofs_dir = os.path.join(previous_optimizer_dir, HOFS_STR, hof_n)
            if os.path.isdir(hofs_dir):
                return hof_n
        raise ValueError("hof nicks are not directories: " + str(hofs_priority))
    else:
        raise ValueError("previous_optimizer_dir is not a directory: " + str(previous_optimizer_dir))
