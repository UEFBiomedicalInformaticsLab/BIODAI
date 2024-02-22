from cross_validation.multi_objective.optimizer.mo_optimizer_including_feature_importance import \
    nick_from_optimizer_and_fi, name_from_optimizer_and_fi
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from input_data.input_data import InputData
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMO
from util.feature_space_lifter import FeatureSpaceLifterMV, FeatureSpaceLifter
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.utils import name_value


class PrefilteredMOOptimizerIncludingFI(MultiObjectiveOptimizer):
    """A pipeline that computes feature importance and then uses it while optimizing."""
    __feature_selector: FeatureSelectorMO
    __feature_importance: MultiViewFeatureImportance
    __optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __optimizer_type: MOOptimizerType

    def __init__(self,
                 feature_importance: MultiViewFeatureImportance,
                 optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                 feature_selector: FeatureSelectorMO):
        self.__feature_selector = feature_selector
        self.__feature_importance = feature_importance
        self.__optimizer = optimizer
        self.__optimizer_type = ConcreteMOOptimizerType(
            uses_inner_models=True,
            nick=nick_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance),
            name=name_from_optimizer_and_fi(optimizer=optimizer, fi=feature_importance))

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                 very_verbose: bool = False
                 ) -> list[MultiObjectiveOptimizerResult]:
        """Make sure we do not include outcomes (potentially affecting feature selection) that are not in objectives."""

        printer.title_print("Computing local active features")
        active_features_by_view = [
            self.__feature_selector.selection_mask(x=v, outcomes=input_data.outcomes(), n_proc=n_proc, printer=printer)
            for v in input_data.views_dict().values()]
        if very_verbose:
            printer.print_variable("number features available", [len(v) for v in active_features_by_view])
            printer.print_variable("number of local active features", [sum(v) for v in active_features_by_view])
        lifter = FeatureSpaceLifterMV(
            single_view_lifters=[FeatureSpaceLifter(active_features=mask) for mask in active_features_by_view])
        lifted_input_data = input_data.uplift(lifter=lifter)
        printer.print("Computing feature importance")
        computed_feature_importance = self.__feature_importance.compute(input_data=lifted_input_data, n_proc=n_proc)
        printer.print("Running optimizer")
        lifted_res = self.__optimizer.optimize_with_feature_importance(
            input_data=lifted_input_data, printer=printer,
            feature_importance=computed_feature_importance,
            n_proc=n_proc,
            workers_printer=workers_printer, logbook_saver=logbook_saver, feature_counts_saver=feature_counts_saver)
        return [r.downlift(lifter) for r in lifted_res]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def __str__(self) -> str:
        res = "Prefiltered optimizer with feature importance\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Feature selector:\n"
        res += str(self.__feature_selector)
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        res += "Multi-view feature importance strategy:\n"
        res += str(self.__feature_importance) + "\n"
        return res
