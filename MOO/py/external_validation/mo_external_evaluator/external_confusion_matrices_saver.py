from collections.abc import Sequence
from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR, \
    save_hof_confusions
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from external_validation.mo_external_evaluator.mo_external_evaluator import MultiObjectiveExternalEvaluator
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer


class ExternalConfusionMatricesSaver(MultiObjectiveExternalEvaluator):
    __objectives: Sequence[PersonalObjective]

    def __init__(self, objectives: list[PersonalObjective]):
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, external_data: InputData, objectives: Sequence[PersonalObjective],
                 optimizer_result: MultiObjectiveOptimizerResult, optimizer_save_path: str, printer: Printer):
        path_saves = optimizer_save_path + "hofs/" + optimizer_result.nick() + "/" + CONFUSION_MATRIX_STR + "/"
        fold_test_data = external_data
        save_hof_confusions(path_saves=path_saves,
                            fold_index=None,
                            hof=optimizer_result,
                            objectives=self.__objectives,
                            fold_test_data=fold_test_data)

    def name(self) -> str:
        return "External confusion matrices saver"
