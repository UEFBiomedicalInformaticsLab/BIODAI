from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import ConfusionMatricesSaver
from cross_validation.multi_objective.cross_evaluator.hypervolume_cross_evaluator import \
    HypervolumeCrossEvaluator
from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import FeatureStabilityMOCrossEval
from cross_validation.multi_objective.cross_evaluator.feature_variety_mo_cross_eval import FeatureVarietyMOCrossEval
from cross_validation.multi_objective.cross_evaluator.hof_saver import HofsSaver
from cross_validation.multi_objective.cross_evaluator.two_objectives_cross_plot import TwoObjectivesCrossPlot
from cross_validation.multi_objective.multi_objective_cross_validation import \
    optimize_with_evaluation, do_final_optimization, create_optimizer_save_path
from setup.allowed_names import DEFAULT_VIEWS
from setup.evaluation_setup import EvaluationSetup
from setup.setup_to_mo_optimizer import setup_to_mo_optimizer
from setup.setup_utils import save_path_from_components, save_config_file, setup_to_folds
from util.printer.printer import Printer
from util.randoms import set_all_seeds


def run_one_setup(setup: EvaluationSetup, printer: Printer,
                  n_workers: int = 1, config_file: str = None):
    set_all_seeds(48723)

    mo_optimizer, input_data, objectives = setup_to_mo_optimizer(setup=setup, printer=printer)

    outer_folds = setup_to_folds(setup=setup, input_data=input_data, printer=printer)
    outer_n_folds = outer_folds.n_folds()
    printer.print_variable("Number of outer folds", outer_n_folds)

    save_path = save_path_from_components(input_data_nick=input_data.nick(),
                                          views_to_use=DEFAULT_VIEWS,
                                          objectives=objectives,
                                          uses_inner_models=mo_optimizer.uses_inner_models(),
                                          outer_n_folds=outer_n_folds)

    optimizer_save_path = create_optimizer_save_path(save_path=save_path, optimizer_nick=mo_optimizer.nick())

    save_config_file(config_file=config_file, destination_path=optimizer_save_path, printer=printer)

    if setup.cross_validation():

        mo_evaluators = [FeatureVarietyMOCrossEval(),
                         FeatureStabilityMOCrossEval(),
                         HofsSaver(save_path=save_path, objectives=objectives),
                         TwoObjectivesCrossPlot(objectives=objectives, save_path=save_path),
                         HypervolumeCrossEvaluator(objectives=objectives),
                         ConfusionMatricesSaver(save_path=save_path, objectives=objectives)
                         ]

        optimize_with_evaluation(
            input_data=input_data, folds=outer_folds,
            mo_optimizer_by_fold=mo_optimizer, mo_evaluators=mo_evaluators,
            save_path=save_path,
            n_proc=n_workers, run_folds_in_parallel=setup.fold_parallelism())

    if setup.final_optimization():
        do_final_optimization(input_data=input_data, mo_optimizer=mo_optimizer.optimizer_for_all_data(),
                              objectives=objectives,
                              save_path=save_path, n_proc=n_workers)
