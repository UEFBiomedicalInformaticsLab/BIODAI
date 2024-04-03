from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import ConfusionMatricesSaver
from cross_validation.multi_objective.cross_evaluator.cross_hypervolume_cross_evaluator import \
    CrossHypervolumeCrossEvaluator
from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import FeatureStabilityMOCrossEval
from cross_validation.multi_objective.cross_evaluator.feature_variety_mo_cross_eval import FeatureVarietyMOCrossEval
from cross_validation.multi_objective.cross_evaluator.folds_saver import FoldsSaver
from cross_validation.multi_objective.cross_evaluator.hof_saver import HofsSaver
from cross_validation.multi_objective.cross_evaluator.two_objectives_cross_plot import TwoObjectivesCrossPlot
from cross_validation.multi_objective.cross_evaluator.used_views import UsedViewsCrossEval
from cross_validation.multi_objective.multi_objective_cross_validation import \
    optimize_with_evaluation, do_final_optimization
from location_manager.location_managers_archive import DEFAULT_LOCATION_MANAGER
from postprocessing.postprocessing import run_postprocessing_archive_cv_and_final
from setup.evaluation_setup import EvaluationSetup
from setup.setup_to_mo_optimizer import setup_to_mo_optimizer
from setup.setup_utils import save_config_file, setup_to_folds
from util.printer.printer import Printer
from util.randoms import set_all_seeds, random_seed
from util.system_utils import cpus_to_use


def run_one_setup(setup: EvaluationSetup, printer: Printer, config_file: str = None):
    seed = setup.seed()
    printer.print("Setting seed " + str(seed))
    set_all_seeds(seed)

    n_workers = cpus_to_use(max_cpus=setup.max_workers(), printer=printer)

    mo_optimizer, input_data, objectives = setup_to_mo_optimizer(setup=setup, printer=printer)

    outer_folds = setup_to_folds(setup=setup, input_data=input_data, printer=printer, seed=random_seed())
    cv_repeats = setup.cv_repeats()
    printer.print_variable("Number of outer folds", str(setup.outer_n_folds())+"x"+str(cv_repeats))

    save_path = DEFAULT_LOCATION_MANAGER.save_path_from_components(
        input_data_nick=input_data.nick(),
        views_to_use=input_data.view_names(),
        objectives=objectives,
        uses_inner_models=mo_optimizer.uses_inner_models(),
        outer_n_folds=setup.outer_n_folds(),
        cv_repeats=cv_repeats,
        setup_seed=seed)

    optimizer_save_path = DEFAULT_LOCATION_MANAGER.optimizer_save_path(
        input_data_nick=input_data.nick(),
        views_to_use=input_data.view_names(),
        objectives=objectives,
        uses_inner_models=mo_optimizer.uses_inner_models(),
        outer_n_folds=setup.outer_n_folds(),
        cv_repeats=cv_repeats,
        optimizer_nick=mo_optimizer.nick(),
        setup_seed=seed)

    save_config_file(config_file=config_file, destination_path=optimizer_save_path, printer=printer)

    if setup.cross_validation():

        mo_evaluators = [
            HofsSaver(save_path=save_path, objectives=objectives),
            ConfusionMatricesSaver(save_path=save_path, objectives=objectives),
            FoldsSaver(optimizer_save_path=optimizer_save_path),
            FeatureVarietyMOCrossEval(),
            FeatureStabilityMOCrossEval(),
            UsedViewsCrossEval(),
            TwoObjectivesCrossPlot(objectives=objectives, save_path=save_path),
            CrossHypervolumeCrossEvaluator(objectives=objectives)
        ]

        optimize_with_evaluation(
            input_data=input_data, folds=outer_folds,
            mo_optimizer_by_fold=mo_optimizer, mo_evaluators=mo_evaluators,
            save_path=save_path,
            n_proc=n_workers, run_folds_in_parallel=setup.fold_parallelism())

        mo_evaluators = None  # Help GC.

    outer_folds = None  # Help GC.

    if setup.final_optimization():
        do_final_optimization(input_data=input_data, mo_optimizer=mo_optimizer.optimizer_for_all_data(),
                              objectives=objectives,
                              save_path=save_path, n_proc=n_workers)

    run_postprocessing_archive_cv_and_final(optimizer_dir=optimizer_save_path, printer=printer)
