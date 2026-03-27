from collections.abc import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.optimizer_utils import individuals_to_hofs
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from individual.individual_with_context import IndividualWithContext
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from input_data.input_data import InputData
from multi_view_utils import collapse_feature_importance_dict
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.recursive_feature_importance import MultiOutcomeRecursiveFeatureImportance
from util.sequence_utils import count_nonzero
from util.sparse_bool_list_by_set import SparseBoolListBySet


class RfeMoOptimizer(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __recursive_fi: MultiOutcomeRecursiveFeatureImportance
    __step: int
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __folds_creator: InputDataFoldsCreator
    __hof_factories: Iterable[HallOfFameFactory]

    __optimizer_type = ConcreteMOOptimizerType(
        uses_inner_models=True, nick="RFE", name="RFE multi-view multi-objective optimizer")

    def __init__(self,
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 recursive_fi: MultiOutcomeRecursiveFeatureImportance,
                 folds_creator: InputDataFoldsCreator,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 step: int = 1):
        self.__recursive_fi = recursive_fi
        self.__step = step
        self.__objectives = objectives
        self.__folds_creator = folds_creator
        self.__hof_factories = hof_factories

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: dict[str,Distribution],
                                         n_proc=1,
                                         workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                                         known_solutions: Sequence[IndividualWithContext] = ()
                                         ) -> Sequence[MultiObjectiveOptimizerResult]:
        """known_solutions are ignored by this algorithm."""
        collapsed_views = input_data.collapsed_views()
        outcomes = input_data.outcomes()
        collapsed_fi = collapse_feature_importance_dict(feature_importance)
        num_selected = count_nonzero(collapsed_fi)

        pop = [self.__distribution_to_individual(distribution=collapsed_fi)]

        while num_selected > 0:
            if num_selected % 10 == 0:
                printer.print(str(num_selected))
            next_to_select = max(0, num_selected-self.__step)
            collapsed_fi = self.__recursive_fi.next_fi(
                x=collapsed_views.to_dataframe(), outcomes=outcomes, feature_importance=collapsed_fi,
                n_features_to_select=next_to_select, n_proc=n_proc)
            individual = self.__distribution_to_individual(distribution=collapsed_fi)
            pop.append(individual)
            num_selected = individual.sum()

        return individuals_to_hofs(input_data=input_data, objectives=self.__objectives,
                                   folds_creator=self.__folds_creator,
                                   pop=pop,
                                   hof_factories=self.__hof_factories,
                                   n_workers=n_proc,
                                   workers_printer=workers_printer)

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def __distribution_to_individual(self, distribution: Distribution) -> PeculiarIndividualSparse:
        if distribution.is_uniform():
            # We map the uniform distribution with the empty individual.
            res_list = SparseBoolListBySet(min_size=len(distribution))
        else:
            res_list = SparseBoolListBySet(seq=distribution.nonzero())
        return PeculiarIndividualSparse(seq=res_list, n_objectives=len(self.__objectives))
