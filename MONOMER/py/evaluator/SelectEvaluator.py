from hyperparam_manager.select_hp_manager import SelectHpManager
from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer, UnbufferedOutPrinter


class SelectEvaluator(WorkersPoolEvaluator):

    def __init__(self, input_data: InputData, hp_manager: SelectHpManager, folds_list,
                 objectives: [PersonalObjective],
                 n_workers=1, seed=876432,
                 workers_printer: Printer = UnbufferedOutPrinter()):
        super().__init__(input_data=input_data, folds_list=folds_list, hp_manager=hp_manager,
                         objectives=objectives,
                         n_workers=n_workers, seed=seed,
                         workers_printer=workers_printer)

    def individual_size(self):
        return self.n_views()
