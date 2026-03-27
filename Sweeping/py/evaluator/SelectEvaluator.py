from collections.abc import Sequence

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.select_mv_hp_manager import SelectMvHpManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import Printer, UnbufferedOutPrinter


class SelectEvaluator(WorkersPoolEvaluator):

    def __init__(self, input_data: InputData, hp_manager: SelectMvHpManager,
                 folds_list: list[tuple[IndexArray,IndexArray]],
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 n_workers=1, seed=876432,
                 workers_printer: Printer = UnbufferedOutPrinter(),
                 compute_confidence: bool = False):
        super().__init__(input_data=input_data, folds_list=folds_list, hp_manager=hp_manager,
                         objectives=objectives,
                         n_workers=n_workers, seed=seed,
                         workers_printer=workers_printer,
                         compute_confidence=compute_confidence)

    def individual_size(self) -> int:
        return self.n_views()