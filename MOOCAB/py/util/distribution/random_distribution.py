import random

from util.distribution.distribution import Distribution, ConcreteDistribution


def random_distribution(n_values: int) -> Distribution:
    """Assigns a random weight to each possible value then normalizes them to sum to 1."""
    return ConcreteDistribution(probs=[random.random() for _ in range(n_values)])