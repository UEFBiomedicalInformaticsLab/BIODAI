import random
import numpy as np


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def random_seed():
    return random.randrange(2**32)
