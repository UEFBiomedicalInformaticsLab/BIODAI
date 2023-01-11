import os
import shutil

from cross_validation.folds import Folds, load_folds
from cross_validation.multi_objective.cross_evaluator.folds_saver import FOLDS_FILE_NAME
from cross_validation.multi_objective.cross_evaluator.hof_saver import HOFS_STR
from cross_validation.multi_objective.optimizer.lasso_mo import LASSO_STR
from folds_creator.input_data_k_folds_creator import InputDataKFoldsCreator
from hall_of_fame.pareto_front import PARETO_NICK
from input_data.brca_input_creator import BrcaInputCreator
from input_data.input_creators_archive import INPUT_CREATORS_DICT
from input_data.input_data import InputData
from input_data.kir_input_creator import KirInputCreator
from input_data.luad_lusc_input_creator import LuadLuscInputCreator
from input_data.prad_input_creator import PradInputCreator
from input_data.swedish_input_creator import SwedishInputCreator
from load_omics_views import MRNA_NAME
from objective.social_objective import PersonalObjective
from setup.allowed_names import STRATIFIED_K_FOLD_NAME, LOAD_FOLD_NAME, AUTO_FOLD_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG, OUTER_N_FOLDS_SMALL
from util.printer.printer import Printer, OutPrinter


def outer_folds_num(use_big_setup: bool) -> int:
    if use_big_setup:
        return OUTER_N_FOLDS_BIG
    else:
        return OUTER_N_FOLDS_SMALL


def create_outer_folds(outer_n_folds: int, input_data: InputData):
    return Folds(test_sets=[f[1] for f in create_outer_folds_list(outer_n_folds=outer_n_folds, input_data=input_data)])


def create_outer_folds_list(outer_n_folds: int, input_data: InputData):
    outer_folds_creator = InputDataKFoldsCreator(n_folds=outer_n_folds)
    return outer_folds_creator.create_folds_from_input_data(input_data=input_data)


def objectives_string(objectives: [PersonalObjective], uses_inner_models: bool) -> str:
    if uses_inner_models:
        objectives_str_list = [o.nick() for o in objectives]
    else:
        objectives_str_list = [o.computer_nick() for o in objectives]
    objectives_str_list.sort()
    objectives_str = ""
    for o in objectives_str_list:
        if objectives_str != "":
            objectives_str += "_"
        objectives_str += o
    return objectives_str


def views_string(view_names: [str]) -> str:
    res = ""
    for n in view_names:
        if res != "":
            res += "_"
        res += str(n)
    return res


def load_input_data(dataset_name: str, views_to_use: [str] = (MRNA_NAME, ), printer: Printer = OutPrinter()) -> InputData:
    if dataset_name == PradInputCreator().nick():
        input_creator = PradInputCreator()
    elif dataset_name == BrcaInputCreator().nick():
        input_creator = BrcaInputCreator()
    elif dataset_name == SwedishInputCreator().nick():
        input_creator = SwedishInputCreator()
    elif dataset_name == LuadLuscInputCreator().nick():
        input_creator = LuadLuscInputCreator()
    elif dataset_name == KirInputCreator().nick():
        input_creator = KirInputCreator()
    else:
        if dataset_name in INPUT_CREATORS_DICT:
            input_creator = INPUT_CREATORS_DICT[dataset_name]
        else:
            raise ValueError("Unknown dataset " + str(dataset_name))
    printer.title_print("Loading " + input_creator.nick() + " cohort")
    res = input_creator.create(views_to_load=views_to_use, printer=printer)
    printer.title_print("Loaded " + str(res.n_samples()) + " samples.")
    return res


def save_path_from_components(input_data_nick: str, views_to_use: [str], objectives: [PersonalObjective],
                              uses_inner_models: bool, outer_n_folds: int) -> str:
    objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
    res = "./" + input_data_nick + "/" + views_string(views_to_use)
    res += "/" + objectives_str + "/" + str(outer_n_folds) + "_folds/"
    return res


def save_path_external_validation(
        input_data_nick: str, views_to_use: [str], objectives: [PersonalObjective],
        uses_inner_models: bool, external_data_nick: str) -> str:
    objectives_str = objectives_string(objectives=objectives, uses_inner_models=uses_inner_models)
    res = "./" + input_data_nick + "/" + views_string(views_to_use)
    res += "/" + objectives_str + "/external_validation/" + external_data_nick + "/"
    return res


def save_config_file(config_file: str, destination_path: str, printer: Printer):
    if config_file is not None:
        printer.print("Copying config file to run directory")
        printer.print_variable("Copy source", config_file)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(config_file, destination_path + "config.ini")


def setup_to_folds(setup: EvaluationSetup, input_data: InputData, printer: Printer) -> Folds:
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
            outer_n_folds = outer_folds_num(use_big_setup=setup.use_big_defaults())
            printer.print("Creating " + str(outer_n_folds) + " stratified folds.")
            outer_folds = create_outer_folds(outer_n_folds=outer_n_folds, input_data=input_data)
            return outer_folds
        elif outer_folds_str == LOAD_FOLD_NAME:
            raise ValueError("Load of folds requested but no valid base directory received.")
        else:
            raise ValueError("Unexpected outer folds setup.")


def choose_hof_nick(previous_optimizer_dir: str, hofs_priority: [str] = (PARETO_NICK, LASSO_STR)) -> str:
    if os.path.isdir(previous_optimizer_dir):
        for hof_n in hofs_priority:
            hofs_dir = os.path.join(previous_optimizer_dir, HOFS_STR, hof_n)
            if os.path.isdir(hofs_dir):
                return hof_n
        raise ValueError("hof nicks are not directories: " + str(hofs_priority))
    else:
        raise ValueError("previous_optimizer_dir is not a directory: " + str(previous_optimizer_dir))
