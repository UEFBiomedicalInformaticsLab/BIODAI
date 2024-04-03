from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from ga_components.feature_counts_saver import DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from individual.individual_with_context import IndividualWithContext
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.single_objective.optimizer.single_objective_optimizer import SOOptimizer
from input_data.input_data import InputData
from util.printer.printer import Printer, NullPrinter


class SOtoMOOptimizerAdapter(MultiObjectiveOptimizer):
    __so_optimizer: SOOptimizer
    __optimizer_type: MOOptimizerType
    __n_objectives: int

    def __init__(self, so_optimizer: SOOptimizer, n_objectives: int):
        self.__so_optimizer = so_optimizer
        self.__optimizer_type =\
            ConcreteMOOptimizerType(
                uses_inner_models=False, nick=so_optimizer.nick(),
                name=so_optimizer.name() + " adapted to multi-objective")
        self.__n_objectives = n_objectives

    def optimize(self, input_data: InputData, printer, n_proc=1,
                 workers_printer: Printer = NullPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver=DummyFeatureCountsSaver()) -> [MultiObjectiveOptimizerResult]:
        """ TODO n_proc is ignored at the moment.
            TODO stratify outcome used as default, allow for other outcomes."""
        so_res = self.__so_optimizer.optimize(views=input_data.views_dict(), y=input_data.stratify_outcome_data())
        return [MultiObjectiveOptimizerResult(
            name="single objective adapted to multi-objective",
            nick="so",
            predictors=[[so_res.predictor]*self.__n_objectives],
            hyperparams=[IndividualWithContext(individual=so_res.hyperparams, hp_manager=so_res.hp_manager)])]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type
