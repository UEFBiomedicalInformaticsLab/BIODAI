from collections import Iterable, Sequence

from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.ga_runner import GARunner
from ga_strategy.ga_strategy_selector import GAStrategySelector
from hyperparam_manager.select_hp_manager import SelectHpManager
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter


class SelectGARunner(GARunner):

    def __init__(self, pop_size, mating_prob, mutation_frequency, hp_manager: SelectHpManager,
                 objectives: Iterable[PersonalObjective],
                 sorting_strategy: SortingStrategy,
                 use_clone_repurposing: bool = False):
        super().__init__(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutation_frequency,
            objectives=objectives, sorting_strategy=sorting_strategy, use_clone_repurposing=use_clone_repurposing)
        self.__hp_manager = hp_manager

    def _create_ga_strategy(self, input_data: InputData, folds_list,
                            feature_importance: Sequence[Distribution] = None,
                            n_workers=1,
                            workers_printer: Printer = UnbufferedOutPrinter()):
        """This strategy does not use feature_importance."""
        return GAStrategySelector(
            input_data=input_data,
            mating_prob=self.mating_prob(), mutation_frequency=self.mutation_frequency(),
            hp_manager=self.__hp_manager, folds_list=folds_list,
            objectives=self._objectives(),
            sorting_strategy=self.sorting_strategy(),
            n_workers=n_workers, workers_printer=workers_printer,
            use_clone_repurposing=self.use_clone_repurposing())
