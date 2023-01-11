from collections.abc import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.guided_forward import GUIDED_FORWARD_OPTIMIZER_TYPE, GuidedForward
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter


class GuidedForwardAcceptingFI(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __objectives: Sequence[PersonalObjective]
    __folds_creator: InputDataFoldsCreator
    __hof_factories: Iterable[HallOfFameFactory]

    def __init__(self,
                 folds_creator: InputDataFoldsCreator,
                 objectives: Sequence[PersonalObjective],
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),)):
        self.__folds_creator = folds_creator
        self.__objectives = objectives
        self.__hof_factories = hof_factories

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: Sequence[Distribution], n_proc=1,
                                         workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                                         ) -> [MultiObjectiveOptimizerResult]:
        optimizer = GuidedForward.create_from_fi(
            feature_importance=feature_importance,
            input_data=input_data,
            objectives=self.__objectives,
            folds_creator=self.__folds_creator,
            hof_factories=self.__hof_factories)
        return optimizer.optimize(
            input_data=input_data, printer=printer, n_proc=n_proc, workers_printer=workers_printer,
            logbook_saver=logbook_saver,
            feature_counts_saver=feature_counts_saver)

    def optimizer_type(self) -> MOOptimizerType:
        return GUIDED_FORWARD_OPTIMIZER_TYPE
