from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from input_data.input_data import InputData
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMO
from util.feature_space_lifter import FeatureSpaceLifterMV, FeatureSpaceLifter
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import set_all_seeds
from util.utils import name_value


class PrefilteredMOOptimizer(MultiObjectiveOptimizer):
    """A pipeline that starts by filtering out some features according to a given filter algorithm."""

    __feature_selector: FeatureSelectorMO
    __optimizer: MultiObjectiveOptimizer

    def __init__(self,
                 feature_selector: FeatureSelectorMO,
                 optimizer: MultiObjectiveOptimizer):
        self.__feature_selector = feature_selector
        self.__optimizer = optimizer

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> [MultiObjectiveOptimizerResult]:
        """Make sure we do not include outcomes (potentially affecting feature selection) that are not in objectives."""

        active_features_by_view = [
            self.__feature_selector.selection_mask(x=v, outcomes=input_data.outcomes(), n_proc=n_proc, printer=printer)
            for v in input_data.views_dict().values()]
        lifter = FeatureSpaceLifterMV(
            single_view_lifters=[FeatureSpaceLifter(active_features=mask) for mask in active_features_by_view])

        lifted_input_data = input_data.uplift(lifter=lifter)

        lifted_res = self.__optimizer.optimize(
            input_data=lifted_input_data, printer=printer, n_proc=n_proc,
            workers_printer=workers_printer, logbook_saver=logbook_saver, feature_counts_saver=feature_counts_saver)

        return [r.downlift(lifter) for r in lifted_res]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer.optimizer_type()

    def __str__(self) -> str:
        res = "Prefiltered optimizer\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += "Feature selector:\n"
        res += str(self.__feature_selector)
        res += "Inner optimizer:\n"
        res += str(self.__optimizer)
        return res
