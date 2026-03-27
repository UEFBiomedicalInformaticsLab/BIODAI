from collections.abc import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.optimizer_utils import individuals_to_hofs
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.mv_feature_set_by_names import MVFeatureSetByNames
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.printer.printer import Printer, UnbufferedOutPrinter


class UserProvidedMoOptimizer(MultiObjectiveOptimizer):
    __feature_sets: Sequence[MVFeatureSetByNames]
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __folds_creator: InputDataFoldsCreator
    __hof_factories: Iterable[HallOfFameFactory]
    __optimizer_type: MOOptimizerType

    def __init__(self,
                 feature_sets: Sequence[MVFeatureSetByNames],
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 folds_creator: InputDataFoldsCreator,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 optimizer_type_nick: str = "user_provided",
                 optimizer_type_name: str = "User provided feature sets multi-view multi-objective fitter"):
        self.__feature_sets = feature_sets
        self.__objectives = objectives
        self.__folds_creator = folds_creator
        self.__hof_factories = hof_factories
        self.__optimizer_type = ConcreteMOOptimizerType(uses_inner_models=True,
                                                        nick=optimizer_type_nick,
                                                        name=optimizer_type_name)

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:
        n_objectives = len(self.__objectives)
        pop = [PeculiarIndividualSparse(
            seq=input_data.get_mask(features_by_names=f), n_objectives=n_objectives) for f in self.__feature_sets]
        return individuals_to_hofs(input_data=input_data,
                                   objectives=self.__objectives,
                                   folds_creator=self.__folds_creator,
                                   pop=pop,
                                   hof_factories=self.__hof_factories,
                                   n_workers=n_proc,
                                   workers_printer=workers_printer)

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type
