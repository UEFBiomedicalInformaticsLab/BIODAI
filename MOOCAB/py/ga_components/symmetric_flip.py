from random import random
from typing import Optional

from individual.Individual import Individual
from util.summable import SummableSequence


def symmetric_flip(individual: Individual, frequency: float):
    """ The frequency of reducing mutations is equal to the frequency of increasing mutations.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
    if frequency < 0.0:
        raise ValueError()
    if frequency == 0.0:
        return individual,

    ind_len = len(individual)
    ind_ones = individual.sum()
    ind_zeros = ind_len - ind_ones
    prob_1_to_0 = min(frequency/(2.0*ind_ones), 1.0)
    prob_0_to_1 = min(frequency/(2.0*ind_zeros), 1.0)

    for i in range(ind_len):
        if individual[i]:
            if random() < prob_1_to_0:
                individual[i] = type(individual[i])(False)
        else:
            if random() < prob_0_to_1:
                individual[i] = type(individual[i])(True)
    return individual,


def symmetric_flip_with_mask(individual: Individual, frequency: float, active_mask: Optional[SummableSequence] = None):
    """ The frequency of reducing mutations is equal to the frequency of increasing mutations.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        Individual is modified in place and also returned.

        :param active_mask: bits that are allowed to flip.
        """
    if active_mask is None:
        return symmetric_flip(individual=individual, frequency=frequency)
    else:
        active_len = active_mask.sum()
        ind_ones = individual.sum()  # Can be faster than builtin sum(individual)
        ind_zeros = active_len - ind_ones
        if ind_ones == 0:
            prob_1_to_0 = 1.0
        else:
            prob_1_to_0 = min(frequency / (2.0 * ind_ones), 1.0)
        if ind_zeros == 0:
            prob_0_to_1 = 1.0
        else:
            prob_0_to_1 = min(frequency / (2.0 * ind_zeros), 1.0)
        for i in range(len(individual)):
            if active_mask[i]:
                ind_i = individual[i]
                if ind_i:
                    if random() < prob_1_to_0:
                        individual[i] = type(ind_i)(False)
                else:
                    if random() < prob_0_to_1:
                        individual[i] = type(ind_i)(True)
    return individual,
