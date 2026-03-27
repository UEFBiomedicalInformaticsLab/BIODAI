from collections.abc import Sequence

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from hyperparam_manager.mv_hyperparam_manager.theta_mv_hp_manager import ThetaMvHpManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import Printer, UnbufferedOutPrinter


class ThetaEvaluator(WorkersPoolEvaluator):

    def __init__(self, input_data: InputData, folds_list, objectives: Sequence[PersonalObjectiveWithImportance],
                 theta: float,
                 n_workers: int = 1, seed: int = 8745,
                 workers_printer: Printer = UnbufferedOutPrinter(),
                 compute_feature_importance: bool = False,
                 compute_confidence: bool = False):
        super().__init__(
            input_data, folds_list=folds_list,
            hp_manager=ThetaMvHpManager(
                theta=theta,
                adj_view_def=input_data.adjusted_view_def(),
                view_col_numbers=input_data.n_features_per_view()),
            objectives=objectives,
            n_workers=n_workers,
            seed=seed, workers_printer=workers_printer,
            compute_feature_importance=compute_feature_importance,
            compute_confidence=compute_confidence)

    def individual_size(self):
        return self.n_predictive_features()