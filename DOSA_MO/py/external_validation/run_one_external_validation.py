from cross_validation.multi_objective.multi_objective_cross_validation import do_final_optimization, \
    create_final_optimizer_printer, VALIDATION_REGISTRY_FILE_NAME
from location_manager.location_managers_archive import DEFAULT_LOCATION_MANAGER
from location_manager.path_utils import create_optimizer_save_path
from external_validation.mo_external_evaluator.cross_hypervolume_external_evaluator import \
    CrossHypervolumeExternalEvaluator
from external_validation.mo_external_evaluator.external_confusion_matrices_saver import ExternalConfusionMatricesSaver
from external_validation.mo_external_evaluator.hof_saver import ExternalHofsSaver
from external_validation.mo_external_evaluator.two_objectives_external_plot import TwoObjectivesExternalPlot
from hall_of_fame.hof_utils import hof_path
from input_data.input_data import select_common_features
from input_data.input_data_utils import select_outcomes_in_objectives
from postprocessing.external_postprocessing import run_postprocessing_archive_external
from setup.evaluation_setup import EvaluationSetup
from setup.setup_to_mo_optimizer import setup_to_mo_optimizer
from setup.setup_utils import load_input_data, save_config_file
from util.printer.printer import Printer
from util.randoms import set_all_seeds
from util.system_utils import cpus_to_use
from validation_registry.validation_registry import FileValidationRegistry


def run_one_external_validation(setup: EvaluationSetup, printer: Printer, config_file: str = None):
    seed = setup.seed()
    printer.print("Setting seed " + str(seed))
    set_all_seeds(seed)

    n_workers = cpus_to_use(max_cpus=setup.max_workers(), printer=printer)

    views_to_use = setup.views_to_use()

    mo_optimizer, input_data, objectives = setup_to_mo_optimizer(setup=setup, printer=printer)

    external_data = load_input_data(dataset_name=setup.external_dataset(), views_to_use=views_to_use, printer=printer)
    printer.print("Removing outcomes not necessary for objectives from external data.")
    external_data = select_outcomes_in_objectives(input_data=external_data, objectives=objectives)

    save_path = DEFAULT_LOCATION_MANAGER.save_path_external_from_strings(
        input_data_nick=input_data.nick(),
        views_to_use=external_data.view_names(),
        objectives=objectives,
        uses_inner_models=mo_optimizer.uses_inner_models(),
        external_data_nick=external_data.nick(),
        setup_seed=seed)

    optimizer_nick = mo_optimizer.nick()
    optimizer_save_path = create_optimizer_save_path(save_path=save_path, optimizer_nick=optimizer_nick)
    printer = create_final_optimizer_printer(
        optimizer_save_path=optimizer_save_path)

    printer.title_print("Reducing datasets to common features.")
    input_data, external_data = select_common_features(a=input_data, b=external_data)
    printer.title_print("Standardizing the features of each dataset separately.")
    input_data = input_data.standardize_features()
    external_data = external_data.standardize_features()
    printer.print("Internal data")
    printer.print(input_data)
    printer.print("External data")
    printer.print(external_data)

    save_config_file(config_file=config_file, destination_path=optimizer_save_path, printer=printer)

    printer.title_print("Optimizer details")
    printer.print(mo_optimizer)

    optimizer_results = do_final_optimization(input_data=input_data,
                                              mo_optimizer=mo_optimizer.optimizer_for_all_data(),
                                              objectives=objectives,
                                              save_path=save_path, n_proc=n_workers)
    printer.title_print("Optimizer results")
    printer.print_in_lines(optimizer_results)

    evaluators = [ExternalHofsSaver(),
                  TwoObjectivesExternalPlot(),
                  ExternalConfusionMatricesSaver(objectives=objectives),
                  CrossHypervolumeExternalEvaluator(objectives=objectives)]

    printer.title_print("Applying evaluators")
    for hof in optimizer_results:
        printer.title_print("Evaluating hall of fame " + hof.name())

        printer.print("Starting a fresh validation registry file.")
        hof_registry = FileValidationRegistry(
            file_path=hof_path(
                optimizer_save_path=optimizer_save_path, hof_nick=hof.nick()) + VALIDATION_REGISTRY_FILE_NAME)
        hof_registry.clean()  # We do not want properties from previous potentially incoherent runs.

        for evaluator in evaluators:
            printer.title_print("Applying " + evaluator.name())
            evaluator.evaluate(
                input_data=input_data,
                external_data=external_data,
                objectives=objectives,
                optimizer_result=hof,
                optimizer_save_path=optimizer_save_path,
                printer=printer,
                hof_registry=hof_registry)

    run_postprocessing_archive_external(optimizer_dir=optimizer_save_path, printer=printer)
