import math

import numpy
from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist
from typing import List, Sequence

from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from hyperparam_manager.hyperparam_manager import HyperparamManager
from util import sparse_bool_list_by_set
from util.math.list_math import list_div
from util.math.summer import KahanSummer


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


def assign_peculiarity(individuals: List[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
    """Assigns a peculiarity to all individuals.
    """
    average = average_individual(individuals=individuals, hp_manager=hp_manager)
    average_sum = KahanSummer.sum(average)
    if average_sum < 0.0:
        raise ValueError("Sum of frequencies of features cannot be negative.")
    elif average_sum == 0.0:
        for i in individuals:
            i.set_peculiarity(1.0)  # If average_sum is 0 we set every peculiarity as 1.
    else:
        for i in individuals:
            i_features = hp_manager.active_features_mask(i)
            dot = sparse_bool_list_by_set.dot_product(average, i_features)
            denominator = max(average_sum, hp_manager.n_active_features(i))
            if denominator == 0.0:
                raise Exception("Denominator is zero, should not happen.\n" +
                                "average_sum: " + str(average_sum) + "\n" +
                                "sum(i_features): " + str(KahanSummer.sum(i_features)) + "\n")
            i.set_peculiarity(1.0 - (dot / denominator))


def assign_just_social_space(individuals: List[PeculiarIndividualByListlike]):
    """Assigns a social space to all individuals. Assuming crowding distance and peculiarity are already computed.
        """

    n_individuals = len(individuals)
    crowd_sum = KahanSummer()
    n_crowd_inf = 0
    max_crowd = 0.0
    peculiarity_sum = KahanSummer()
    inf_substitute = 1.0
    # Default value in case there are no infinities or only infinities, to avoid warnings when multiplying later.

    for i in individuals:
        peculiarity_sum.add(i.get_peculiarity())
        crowd_dist = i.get_crowding_distance()
        if math.isinf(crowd_dist):
            n_crowd_inf += 1
        else:
            max_crowd = max(max_crowd, crowd_dist)
            crowd_sum.add(crowd_dist)
    peculiarity_sum_val = peculiarity_sum.get_sum()
    if max_crowd > 0.0:
        inf_substitute = max_crowd * 2  # We consider infinities as double the maximum of the others.
    crowd_sum.add(n_crowd_inf * inf_substitute)
    crowd_sum_val = crowd_sum.get_sum()
    if crowd_sum_val > 0:
        crowd_mult = n_individuals / (crowd_sum_val*2)  # Multiplier to produce a mean of 0.5
    else:
        crowd_mult = 1
    if peculiarity_sum_val > 0:
        pec_mult = n_individuals / (peculiarity_sum_val*2)
    else:
        pec_mult = 1
    for i in individuals:
        crowd_dist = i.get_crowding_distance()
        if math.isinf(crowd_dist):
            crowd_dist = inf_substitute
        peculiarity = i.get_peculiarity()
        social_space = crowd_dist * crowd_mult + peculiarity * pec_mult
        if not social_space >= 0.0:
            raise ValueError("Social space is not valid\n" +
                             "crowd dist: " + str(crowd_dist) + "\n" +
                             "peculiarity: " + str(peculiarity) + "\n" +
                             "crowd_mult: " + str(crowd_mult) + "\n" +
                             "pec_mult: " + str(pec_mult) + "\n")
        i.set_social_space(social_space)


def assign_social_space(individuals: List[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
    """Assigns a social space to all individuals. Assigns also a new crowd distance and peculiarity as side effects.
        """
    assign_crowd_dist_all_fronts(individuals)
    assign_peculiarity(individuals, hp_manager=hp_manager)
    assign_just_social_space(individuals=individuals)
