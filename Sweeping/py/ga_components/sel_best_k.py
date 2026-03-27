from functools import cmp_to_key

from comparators import Comparator


def sel_best_k(individuals, k, comparator: Comparator):
    """Selects the best k according to the comparator.
    """

    temp_individuals = sorted(individuals, key=cmp_to_key(comparator.compare))

    return list(temp_individuals[:k])
    # Creating a new list since slicing keeps the whole list from being garbage collected.
