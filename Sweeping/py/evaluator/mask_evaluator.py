from collections.abc import Sequence

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import Printer, UnbufferedOutPrinter


class MaskEvaluator(WorkersPoolEvaluator):

    def __init__(self, input_data: InputData, folds_list: list[tuple[IndexArray,IndexArray]],
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 n_workers: int = 1, seed: int = 8745,
                 workers_printer: Printer = UnbufferedOutPrinter(),
                 compute_feature_importance: bool = False,
                 compute_confidence: bool = False):
        hp_manager = MaskMvHpManager.create_from_input_data(input_data=input_data)
        super().__init__(
            input_data, folds_list=folds_list, hp_manager=hp_manager,
            objectives=objectives,
            n_workers=n_workers,
            seed=seed, workers_printer=workers_printer,
            compute_feature_importance=compute_feature_importance,
            compute_confidence=compute_confidence)

    def individual_size(self):
        return self.n_predictive_features()
