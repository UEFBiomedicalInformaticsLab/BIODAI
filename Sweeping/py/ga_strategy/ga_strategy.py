from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Callable

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from ga_components.sorter.sorting_strategy import SortingStrategy
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.Individual import Individual
from individual.peculiar_individual import PeculiarIndividual
from objective.social_objective import PersonalObjective


class GAStrategy(ABC):
    __evaluator: WorkersPoolEvaluator
    __objectives: Sequence[PersonalObjective]
    __sorting_strategy: SortingStrategy
    __mating_prob: float
    __mutating_prob: float
    __use_clone_repurposing: bool

    def __init__(self, evaluator: WorkersPoolEvaluator, objectives: Iterable[PersonalObjective],
                 mating_prob: float, mutation_frequency: float, sorting_strategy: SortingStrategy,
                 use_clone_repurposing: bool = False):
        self.__evaluator = evaluator
        self.__mating_prob = mating_prob
        self.__mutating_prob = mutation_frequency
        self.__objectives = list(objectives)
        self.__sorting_strategy = sorting_strategy
        self.__use_clone_repurposing = use_clone_repurposing

    @abstractmethod
    def create_individual(self) -> PeculiarIndividual:
        raise NotImplementedError()

    @abstractmethod
    def mate(self, ind1, ind2):
        raise NotImplementedError()

    @abstractmethod
    def mutate(self, individual: Individual) -> tuple[Individual]:
        """Individual is mutated in place and also returned. Returns a tuple containing one individual
        for DEAP compatibility."""
        raise NotImplementedError()

    def hp_manager(self) -> MvHyperparamManager:
        return self.evaluator().hp_manager()

    def sorting_strategy_before_selection(self, individuals) -> Sequence[PeculiarIndividual]:
        return self.__sorting_strategy.apply_before_selection(pop=individuals, hp_manager=self.hp_manager())

    def sorting_strategy_after_selection(self, individuals) -> Sequence[PeculiarIndividual]:
        return self.__sorting_strategy.apply_after_selection(pop=individuals, hp_manager=self.hp_manager())

    def tournament(self, pop: Sequence[PeculiarIndividual], k: int) -> Sequence[PeculiarIndividual]:
        return self.__sorting_strategy.tournament(pop=pop, k=k)

    def clone_repurposing(self, pop: list[PeculiarIndividual]) -> Sequence[PeculiarIndividual]:
        """ Passed pop might be changed in place."""
        if self.__use_clone_repurposing:
            hp_man = self.hp_manager()
            uniques = set()
            for i in range(len(pop)):
                ind_key = hp_man.to_tuple(hyperparams=pop[i])
                if ind_key in uniques:
                    pop[i] = self.create_individual()
                else:
                    uniques.add(ind_key)
        return pop

    def mating_prob(self) -> float:
        """The probability of applying crossover operator."""
        return self.__mating_prob

    def mutation_frequency(self) -> float:
        """The frequency of mutation of features for each individual. If an individual has N features,
        the probability of mutation of each feature is mutation_frequency/N."""
        return self.__mutating_prob

    def individual_size(self) -> int:
        """Actual number of attributes of an individual. May be different from the number of selected features.
        Used when creating a new individual to generate its attributes."""
        return self.__evaluator.individual_size()

    def evaluator(self):
        return self.__evaluator

    def objectives(self):
        return self.__objectives

    def to_be_added_to_stats(self) -> dict[str, Callable[[PeculiarIndividual], float]]:
        return self.__sorting_strategy.to_be_added_to_stats()

    def n_objectives(self) -> int:
        return len(self.__objectives)

    def select(self, pop: Sequence[PeculiarIndividual], pop_size: int) -> Sequence[PeculiarIndividual]:
        return self.__sorting_strategy.select(pop=pop, pop_size=pop_size)
