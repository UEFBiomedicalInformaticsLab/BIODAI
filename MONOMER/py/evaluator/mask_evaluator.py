from collections.abc import Sequence

from hyperparam_manager.dummy_hp_manager import DummyHpManager
from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer, UnbufferedOutPrinter


class MaskEvaluator(WorkersPoolEvaluator):

    def __init__(self, input_data: InputData, folds_list, objectives: Sequence[PersonalObjective],
                 n_workers: int = 1, seed: int = 8745,
                 workers_printer: Printer = UnbufferedOutPrinter()):
        super().__init__(
            input_data, folds_list=folds_list, hp_manager=DummyHpManager(),
            objectives=objectives,
            n_workers=n_workers,
            seed=seed, workers_printer=workers_printer)

    def individual_size(self):
        return self.n_features()
