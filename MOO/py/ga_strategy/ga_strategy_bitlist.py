import random
from collections.abc import Iterable
from typing import Sequence

from evaluator.mask_evaluator import MaskEvaluator
from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from individual.num_features import NumFeatures
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from ga_strategy.ga_strategy import GAStrategy

from individual.sparse_individual import SparseIndividual
from input_data.input_data import InputData
from multi_view_utils import collapse_feature_importance
from objective.social_objective import PersonalObjective
from util.cached_tuple import CachedTuple
from util.distribution.distribution import Distribution
from util.distribution.uniform_distribution import UniformDistribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.sparse_bool_list_by_set import SparseBoolListBySet


def cx_uniform(ind1: SparseIndividual, ind2: SparseIndividual, indpb):
    """Custom version of the cxUniform of DEAP.

    This version leverages the sparse lists to improve performance.

    Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probability for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in list(ind1.true_positions()):  # Copied in list to avoid changing the set while it is iterated.
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    for i in list(ind2.true_positions()):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


class GAStrategyBitlist(GAStrategy):
    __active_features: CachedTuple[bool]
    __initial_features: NumFeatures
    __collapsed_feature_importance: Distribution  # Potentially could be used also in mutation.
    __non_zero_fi_num: int
    __mutation: BitlistMutation

    def __init__(self, input_data: InputData, mating_prob: float, mutation_frequency: float,
                 initial_features: NumFeatures,
                 folds_list,
                 objectives: Iterable[PersonalObjective],
                 sorting_strategy: SortingStrategy,
                 feature_importance: Sequence[Distribution] = None,
                 n_workers=1,
                 active_features: Sequence[bool] = None, workers_printer: Printer = UnbufferedOutPrinter(),
                 mutation: BitlistMutation = FlipMutation(),
                 use_clone_repurposing: bool = False):
        super().__init__(
            evaluator=MaskEvaluator(input_data=input_data, folds_list=folds_list,
                                    objectives=objectives,
                                    n_workers=n_workers,
                                    workers_printer=workers_printer),
            objectives=objectives,
            mating_prob=mating_prob,
            mutation_frequency=mutation_frequency,
            sorting_strategy=sorting_strategy,
            use_clone_repurposing=use_clone_repurposing
        )
        self.__initial_features = initial_features
        if active_features is None:
            self.__active_features = CachedTuple([1] * self.evaluator().n_features())
        else:
            self.__active_features = CachedTuple(active_features)
        if feature_importance is None:
            self.__collapsed_feature_importance = UniformDistribution(size=self.evaluator().n_features()).as_cached()
        else:
            self.__collapsed_feature_importance = collapse_feature_importance(
                distributions=feature_importance).as_cached()
        self.__mutation = mutation
        self.__non_zero_fi_num = sum([x > 0.0 for x in self.__collapsed_feature_importance])

    def create_individual(self) -> PeculiarIndividualSparse:
        ind_size = self.individual_size()
        res_list = SparseBoolListBySet(min_size=ind_size)
        n_true_bits = self.__initial_features.extract(ind_size)
        for i in range(0, min(n_true_bits, self.__non_zero_fi_num)):
            proceed = True
            while proceed:
                extracted = self.__collapsed_feature_importance.extract()
                if self.__active_features[extracted] and not res_list[extracted]:
                    res_list[extracted] = True
                    proceed = False
        return PeculiarIndividualSparse(seq=res_list, n_objectives=self.n_objectives())

    def mate(self, ind1, ind2):
        return cx_uniform(ind1=ind1, ind2=ind2, indpb=0.5)
        # indpb: independent probability for each attribute to be exchanged

    def mutate(self, individual):
        return self.__mutation.mutate(
            individual=individual, frequency=self.mutation_frequency(), active_mask=self.__active_features)
