import numpy
from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist
from typing import List, Sequence

from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from hyperparam_manager.hyperparam_manager import HyperparamManager
from util import sparse_bool_list_by_set
from util.list_math import list_div


def assign_crowd_dist_all_fronts(individuals: List[PeculiarIndividualByListlike]):
    """Assigns a crowding distance to all individuals taking fronts into account as per NSGA2.
    """
    k = len(individuals)
    pareto_fronts = sortNondominated(individuals=individuals, k=k)

    for front in pareto_fronts:
        assignCrowdingDist(front)


def sum_of_individuals(individuals: Sequence[PeculiarIndividualByListlike],
                       hp_manager: HyperparamManager = DummyHpManager()) -> numpy.ndarray:
    if len(individuals) == 0:
        return numpy.zeros(0, numpy.int)
    else:
        tot_features = None
        for i in individuals:
            i_features = hp_manager.active_features_mask(i)
            if tot_features is None:
                tot_features = i_features
            else:
                tot_features = sparse_bool_list_by_set.add(tot_features, i_features)
                # Should be stable numerically since it is a list of integer counters.
        return tot_features


def feature_distribution(individuals: List[PeculiarIndividualByListlike],
                         hp_manager: HyperparamManager = DummyHpManager()) -> numpy.ndarray:
    if len(individuals) == 0:
        return numpy.zeros(0, numpy.int)
    tot_features = sum_of_individuals(individuals, hp_manager)
    return list_div(tot_features, sum(tot_features))


def average_individual(individuals: Sequence[PeculiarIndividualByListlike],
                       hp_manager: HyperparamManager = DummyHpManager()) -> numpy.ndarray:
    if len(individuals) == 0:
        return numpy.zeros(0, numpy.int)
    tot_features = sum_of_individuals(individuals, hp_manager)
    return list_div(tot_features, len(individuals))  # This is the average individual.
