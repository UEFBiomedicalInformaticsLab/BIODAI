import contextlib
import hashlib
import random
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from util.printer.printer import NULL_PRINTER, Printer


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


def _combined_bytes(a, b) -> bytes:
    return (str(a) + str(b)).encode('utf-8')


def seed_from_objects(a, b = 0) -> int:
    b = _combined_bytes(a, b)
    hash_object = hashlib.sha256(b)
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    seed = hash_int % (2**32)  # To be in the valid range for np.random.seed
    return seed



class SeedIterator(Iterator[int]):
    __seeding_str: str
    __next_i: int

    def __init__(self, seeding_str: str):
        self.__seeding_str = seeding_str
        self.__next_i = 0

    def combined_seed(self) -> bytes:
        return _combined_bytes(self.__seeding_str, str(self.__next_i))

    def __next__(self) -> int:
        seed = seed_from_objects(a=self.__seeding_str, b=self.__next_i)
        self.__next_i += 1
        return seed

    def set_all_seeds(self):
        seed = next(self)
        set_all_seeds(seed=seed)

    def __str__(self) -> str:
        return str(self.combined_seed())


class SeedIterable(Iterable[int]):
    __seeding_str: str

    def __init__(self, seeding_object: Any = None):
        """If seeding_object is None, the current random state is used for seeding."""
        if seeding_object is None:
            seeding_object = random.getstate()
        self.__seeding_str = str(seeding_object)

    def __iter__(self) -> SeedIterator:
        return SeedIterator(self.__seeding_str)

    def __str__(self) -> str:
        return str(self.__seeding_str)


@contextlib.contextmanager
def random_state_context(seed: Any = 98542, printer: Printer = NULL_PRINTER):
    """Saves the current random state and restores it at the end of the block.
    Inside the block it uses a new random state that will depend on the seed passed as a parameter.
    Example of use:
    with random_state_context(seed=87432, printer=printer):"""
    saved_state = random.getstate()
    saved_state_np = np.random.get_state()
    seed = seed_from_objects(a=seed)
    if not printer.is_null():
        printer.print("Setting seed to " + str(seed))
    set_all_seeds(seed=seed)

    try:
        yield
    finally:
        # Restore the saved random state
        if not printer.is_null():
            printer.print(
                "Resetting the random state. Hash value: " + hashlib.sha256(str(saved_state).encode()).hexdigest())
        random.setstate(saved_state)
        if not printer.is_null():
            printer.print(
                "Resetting the Numpy random state. Hash value: " +
                str(hashlib.sha256(str(saved_state_np).encode()).hexdigest()))
        np.random.set_state(saved_state_np)


@contextlib.contextmanager
def random_state_context_blended(additional_seed: Any = 0, printer: Printer = NULL_PRINTER):
    """Saves the current random state and restores it at the end of the block.
    Inside the block it uses a new random state that will depend on the previous random state but also on an
    additional seed passed as a parameter.
    The additional seed is meant to reduce the probability of clashes with other parts of the pipeline,
    that would lead to a repeated sequence of generated pseudorandom values.
    Example of use:
    with random_state_context_blended(additional_seed=87432, printer=printer):"""
    saved_state = random.getstate()
    saved_state_np = np.random.get_state()
    seed = seed_from_objects(a=random_seed(), b=additional_seed)
    printer.print("Setting seed to " + str(seed))
    set_all_seeds(seed=seed)

    try:
        yield
    finally:
        # Restore the saved random state
        printer.print(
            "Resetting the random state. Hash value: " + hashlib.sha256(str(saved_state).encode()).hexdigest())
        random.setstate(saved_state)
        printer.print(
            "Resetting the Numpy random state. Hash value: " +
            str(hashlib.sha256(str(saved_state_np).encode()).hexdigest()))
        np.random.set_state(saved_state_np)