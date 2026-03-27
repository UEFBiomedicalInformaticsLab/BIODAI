from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from feature_importance.multi_outcome_feature_importance import MultiOutcomeFeatureImportance
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from input_data.input_data import InputData
from util.named import NickNamed
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.str_utils import name_value
from view_adjuster.adjust_input_data import adjust_input_data


def nick_from_optimizer_and_fi_nicks(optimizer: str, fi: str) -> str:
    return optimizer + "_" + fi


def nick_from_optimizer_and_fi(optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                               fi: NickNamed) -> str:
    return nick_from_optimizer_and_fi_nicks(optimizer=optimizer.nick(), fi=fi.nick())


def name_from_optimizer_and_fi(optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                               fi: NickNamed) -> str:
    return optimizer.name() + " with " + fi.name()


class MOOptimizerIncludingFeatureImportance(MultiObjectiveOptimizer):
    """A pipeline that computes feature importance and then uses it while optimizing."""

    __feature_importance: MultiOutcomeFeatureImportance
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __optimizer_type: MOOptimizerType

    def __init__(self, feature_importance: MultiOutcomeFeatureImportance,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance):
        self.__feature_importance = feature_importance
        self.__optimizer = optimizer
        self.__optimizer_type = ConcreteMOOptimizerType(
            uses_inner_models=True,
            nick=nick_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance),
            name=name_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance))

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:

        printer.title_print("Adjusting input data for computing feature importance")
        adjusted_input_data = adjust_input_data(input_data=input_data, printer=printer)
        feature_importance = self.__feature_importance.compute(
            input_data=adjusted_input_data, n_proc=n_proc, printer=printer)
        for k,v in feature_importance.items():
            printer.print(k + " feature importance plot:\n" + v.ascii_plot() + "\n")
        return self.__optimizer.optimize_with_feature_importance(
            input_data=input_data, printer=printer,
            feature_importance=feature_importance,
            n_proc=n_proc,
            workers_printer=workers_printer, logbook_saver=logbook_saver, feature_counts_saver=feature_counts_saver)

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def __str__(self) -> str:
        res = "Optimizer with feature importance\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        res += "Multi-view feature importance strategy:\n"
        res += str(self.__feature_importance) + "\n"
        return res
