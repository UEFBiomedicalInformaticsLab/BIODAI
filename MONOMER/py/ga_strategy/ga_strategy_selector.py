import random

from deap import tools

from evaluator.SelectEvaluator import SelectEvaluator
from ga_components.sorter.sorting_strategy import SortingStrategy
from individual.peculiar_individual_dense import PeculiarIndividualDense
from ga_strategy.ga_strategy import GAStrategy
from hyperparam_manager.select_hp_manager import SelectHpManager
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer, UnbufferedOutPrinter


class GAStrategySelector(GAStrategy):

    def __init__(
            self, input_data: InputData, mating_prob, mutation_frequency, hp_manager: SelectHpManager, folds_list,
            objectives: [PersonalObjective],
            sorting_strategy: SortingStrategy,
            n_workers=1,
            workers_printer: Printer = UnbufferedOutPrinter(),
            use_clone_repurposing: bool = False):
        super().__init__(SelectEvaluator(input_data=input_data, hp_manager=hp_manager,
                                         objectives=objectives,
                                         folds_list=folds_list,
                                         n_workers=n_workers, workers_printer=workers_printer),
                         objectives=objectives,
                         mating_prob=mating_prob, mutation_frequency=mutation_frequency,
                         sorting_strategy=sorting_strategy,
                         use_clone_repurposing=use_clone_repurposing)
        self.__hp_manager = hp_manager

    def create_individual(self) -> PeculiarIndividualDense:
        res = PeculiarIndividualDense(n_objectives=self.n_objectives())
        for i in range(0, self.individual_size()):
            res.append(random.randint(0, self.__hp_manager.max_view_individual_index(i)))
        return res

    def mate(self, ind1, ind2):
        return tools.cxUniform(ind1=ind1, ind2=ind2, indpb=0.5)
        # indpb: independent probability for each attribute to be exchanged

    def mutate(self, individual):
        indpb = self.mutation_frequency() / self.individual_size()
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(0, self.__hp_manager.max_view_individual_index(i))
        return individual,
