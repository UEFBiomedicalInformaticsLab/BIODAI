from collections.abc import Iterable

from util.sequence_utils import same_len


def check_none(x):
    if x is None:
        raise ValueError("Unexpected None")
    return x


def check_same_len(a: Iterable, b: Iterable):
    """Uses the len method if possible, otherwise iterates."""
    if not same_len(a=a, b=b):
        raise ValueError("The iterables do not have the same length.")
