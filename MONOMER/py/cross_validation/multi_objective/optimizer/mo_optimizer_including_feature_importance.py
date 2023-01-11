from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer, \
    MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from input_data.input_data import InputData
from util.named import NickNamed
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.utils import name_value


def nick_from_optimizer_and_fi(optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                               fi: NickNamed) -> str:
    return optimizer.nick() + "_" + fi.nick()


def name_from_optimizer_and_fi(optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                               fi: NickNamed) -> str:
    return optimizer.name() + " with " + fi.name()


class MOOptimizerIncludingFeatureImportance(MultiObjectiveOptimizer):
    """A pipeline that computes feature importance and then uses it while optimizing."""

    __feature_importance: MultiViewFeatureImportance
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __optimizer_type: MOOptimizerType

    def __init__(self, feature_importance: MultiViewFeatureImportance,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance):
        self.__feature_importance = feature_importance
        self.__optimizer = optimizer
        self.__optimizer_type = ConcreteMOOptimizerType(
            uses_inner_models=True,
            nick=nick_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance),
            name=name_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance))

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()) -> MultiObjectiveOptimizerResult:
        return self.__optimizer.optimize_with_feature_importance(
            input_data=input_data, printer=printer,
            feature_importance=self.__feature_importance.compute(input_data=input_data, n_proc=n_proc),
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
