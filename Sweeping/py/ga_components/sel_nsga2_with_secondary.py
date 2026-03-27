from functools import cmp_to_key
from itertools import chain

from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist

from comparators import Comparator


def sel_nsga2_with_secondary(individuals, k: int, secondary_comparator: Comparator):
    """Modified from DEAP. This version allows for a custom secondary comparator.

    Apply NSGA-II selection operator on the *individuals* with a specified secondary sorting possibly different
    from the default secondary sorting by crowding distance.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param secondary_comparator: A secondary comparator to sort inside a front.
    :returns: A list of selected individuals.
    """
    pareto_fronts = sortNondominated(individuals, k)

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=cmp_to_key(secondary_comparator.compare))
        chosen.extend(sorted_front[:k])

    return chosen
