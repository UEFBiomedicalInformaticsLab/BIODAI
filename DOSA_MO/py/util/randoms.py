import random

import numpy as np


def random_seed() -> int:
    """Uses module random to extract a new seed."""
    return random.randrange(2**32)


def set_all_seeds(seed: int = 42):
    """Sets the seed in both random and numpy.random.
    Warning: sets the same seed in both random states,
    so using both of them can potentially produce correlated results."""
    random.seed(seed)
    np.random.seed(seed)


def log10_random(min_val: float, max_val: float) -> float:
    """Returns a number with uniform distribution on base 10 logarithmic scale."""
    return 10 ** random.uniform(np.log10(min_val), np.log10(max_val))
