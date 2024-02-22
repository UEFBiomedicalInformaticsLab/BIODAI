from typing import Sequence

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.fronts import PARETO_NICK
from individual.individual_with_context import IndividualWithContext
from input_data.input_data import InputData
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter


class MOPipe(MultiObjectiveOptimizerAcceptingFeatureImportance):
    """Warning: this class has not been tested yet."""
    __optimizers: Sequence[MultiObjectiveOptimizerAcceptingFeatureImportance]
    __optimizer_type: MOOptimizerType
    __piped_hof_nick: str

    def __init__(self, optimizers: Sequence[MultiObjectiveOptimizerAcceptingFeatureImportance],
                 piped_hof_nick: str = PARETO_NICK):
        self.__optimizers = optimizers
        self.__piped_hof_nick = piped_hof_nick
        uses_inner = False
        nick = ""
        name = "pipeline of "
        first = True
        for o in optimizers:
            if o.uses_inner_models():
                uses_inner = True
            if not first:
                nick += "_"
                name += " then "
            first = False
            nick += o.nick()
            name += o.name()
        self.__optimizer_type = ConcreteMOOptimizerType(uses_inner_models=uses_inner, nick=nick, name=name)

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: Sequence[Distribution], n_proc=1,
                                         workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                                         known_solutions: Sequence[IndividualWithContext] = ()
                                         ) -> Sequence[MultiObjectiveOptimizerResult]:
        results = []
        first = True
        for o in self.__optimizers:
            if first:
                first = False
            else:
                for r in results:
                    if r.nick() == self.__piped_hof_nick:
                        known_solutions = r.hyperparams()
            results = o.optimize_with_feature_importance(
                input_data=input_data, printer=printer, feature_importance=feature_importance, n_proc=n_proc,
                workers_printer=workers_printer, logbook_saver=logbook_saver,
                feature_counts_saver=feature_counts_saver, known_solutions=known_solutions)
        return results

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type
