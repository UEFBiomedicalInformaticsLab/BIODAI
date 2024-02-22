from abc import abstractmethod, ABC
from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from individual.individual_with_context import IndividualWithContext
from input_data.input_data import InputData
from util.distribution.distribution import Distribution
from util.distribution.uniform_distribution import UniformDistribution
from util.printer.printer import Printer, UnbufferedOutPrinter


class MultiObjectiveOptimizerAcceptingFeatureImportance(MultiObjectiveOptimizer, ABC):

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:
        """This is for optimizing without using feature importance. Uses uniform F.I."""
        views = input_data.views_dict()
        return self.optimize_with_feature_importance(
            input_data=input_data, printer=printer,
            feature_importance=[UniformDistribution(views[v].shape[1]) for v in views],
            n_proc=n_proc, workers_printer=workers_printer,
            logbook_saver=logbook_saver,
            feature_counts_saver=feature_counts_saver)

    @abstractmethod
    def optimize_with_feature_importance(
            self, input_data: InputData, printer: Printer, feature_importance: Sequence[Distribution], n_proc=1,
            workers_printer=UnbufferedOutPrinter(),
            logbook_saver: LogbookSaver = DummyLogbookSaver(),
            feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
            known_solutions: Sequence[IndividualWithContext] = ()) -> Sequence[MultiObjectiveOptimizerResult]:
        """known_solutions can be used to pass previous solutions for a warm start."""
        raise NotImplementedError()
