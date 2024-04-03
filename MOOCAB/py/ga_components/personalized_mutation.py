from typing import Optional
from random import random

from individual.Individual import Individual
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from util.distribution.distribution import ConcreteDistribution
from util.distribution.uniform_distribution import UniformDistribution
from util.math.list_math import list_reciprocal, add_to_all
from util.summable import SummableSequence
from util.math.summer import KahanSummer


TO_ADD_MULTIPLIER = 0.1


def personalized_mutation(
        individual: Individual, frequency: float, active_mask: Optional[SummableSequence] = None) -> tuple[Individual]:
    """ The frequency of reducing mutations is equal to the frequency of increasing mutations.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        Individual is modified in place and also returned.

        :param active_mask: bits that are allowed to flip.
        """
    if active_mask is None:
        raise NotImplementedError()
    else:
        if isinstance(individual, PeculiarIndividualByListlike) and individual.has_personalized_feature_importance():
            ind_ones = individual.sum()  # Can be faster than builtin sum(individual)
            personalized_importance = individual.get_personalized_feature_importance()
            if ind_ones != len(personalized_importance):
                raise ValueError(
                    "Number of active features differs from length of the importance distribution: " +
                    str(ind_ones) + " vs " + str(len(personalized_importance)) + "\n" +
                    "Individual: " + str(individual) + "\n" +
                    "Personalized importance: " + str(personalized_importance) + "\n")
            imp_sum = KahanSummer.sum(personalized_importance)
            if imp_sum > 0:
                to_add = TO_ADD_MULTIPLIER * imp_sum / ind_ones
                distribution_weights = list_reciprocal(add_to_all(personalized_importance, to_add))
                # print("distribution_weights: " + str(distribution_weights))
                # We add a small amount to all so that the smaller importance one does not get selected automatically.
                importance_distribution = ConcreteDistribution(probs=distribution_weights)
            else:
                importance_distribution = UniformDistribution(size=ind_ones)
            active_len = active_mask.sum()
            ind_zeros = active_len - ind_ones
            if ind_zeros == 0:
                prob_0_to_1 = 1.0
            else:
                prob_0_to_1 = min(frequency / (2.0 * ind_zeros), 1.0)
            prob_1_to_0_mult = frequency / 2.0
            dist_pos = 0
            for i in range(len(individual)):
                if active_mask[i]:
                    ind_i = individual[i]
                    if ind_i:
                        if random() < (importance_distribution[dist_pos] * prob_1_to_0_mult):
                            individual[i] = type(ind_i)(False)
                            # print("Importances " + str(personalized_importance))
                            # print("importance_distribution: " + str(importance_distribution))
                            # print("Importances average " + str(KahanSummer.mean(personalized_importance)))
                            # print("Switching to zero at solution pos " + str(dist_pos))
                            # print("With importance " + str(personalized_importance[dist_pos]))
                        dist_pos += 1
                    else:
                        if random() < prob_0_to_1:
                            individual[i] = type(ind_i)(True)
            return individual,
        else:
            raise ValueError(str(individual))
