import copy
import random
from collections.abc import Sequence

from individual.Individual import Individual


def varAnd(population: Sequence[Individual], toolbox, cxpb: float, mutpb: float = 1.0):
    """Custom version of the varAnd of DEAP.

    Uses random.random() for randomness.

    This custom version does not remove fitness values if an individual remains unchanged.

    Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """

    offspring = [copy.deepcopy(ind) for ind in population]

    n_offsprings = len(offspring)

    # Apply crossover and mutation on the offspring
    for i in range(1, n_offsprings, 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])

    for i in range(n_offsprings):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])

    for i in range(n_offsprings):
        if offspring[i] != population[i]:  # In case it has not been mated and the mutation had no effect.
            del offspring[i].fitness.values

    return offspring
