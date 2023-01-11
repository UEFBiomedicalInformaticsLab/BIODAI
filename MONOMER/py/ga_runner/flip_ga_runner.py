from collections import Iterable
from typing import Sequence

from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.ga_runner import GARunner
from ga_strategy.ga_strategy_bitlist import GAStrategyBitlist
from individual.num_features import NumFeatures
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter


class FlipGARunner(GARunner):
    __active_features: list[bool]
    __initial_features: NumFeatures
    __mutation: BitlistMutation

    def __init__(self, pop_size, mating_prob: float, mutation_frequency: float,
                 initial_features: NumFeatures,
                 objectives: Iterable[PersonalObjective],
                 sorting_strategy: SortingStrategy,
                 active_features: list[bool] = None,
                 mutation: BitlistMutation = FlipMutation(),
                 use_clone_repurposing: bool = False):
        super().__init__(pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutation_frequency,
                         objectives=objectives, sorting_strategy=sorting_strategy,
                         use_clone_repurposing=use_clone_repurposing)
        self.__initial_features = initial_features
        self.__active_features = active_features
        self.__mutation = mutation

    def _create_ga_strategy(self, input_data: InputData, folds_list,
                            feature_importance: Sequence[Distribution] = None,
                            n_workers=1,
                            workers_printer: Printer = UnbufferedOutPrinter()):
        return GAStrategyBitlist(
            input_data=input_data, mating_prob=self.mating_prob(), mutation_frequency=self.mutation_frequency(),
            initial_features=self.__initial_features, folds_list=folds_list,
            objectives=self._objectives(),
            sorting_strategy=self.sorting_strategy(),
            n_workers=n_workers,
            active_features=self.__active_features,
            feature_importance=feature_importance,
            workers_printer=workers_printer,
            mutation=self.__mutation,
            use_clone_repurposing=self.use_clone_repurposing()
        )
