from __future__ import annotations
from collections.abc import Sequence
from typing import Iterable
from typing import Protocol, TypeVar, Iterator


T = TypeVar("T")


class SizedIterable(Protocol[T]):
    """Every class implementing the following methods will be considered a SizedIterable."""
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...


def indices_of(iterable: Iterable, target) -> Sequence[int]:
    """
    Return a list of indices where `target` appears in `iterable`.
    Works for any iterable (lists, tuples, generators, etc.).
    """
    return [i for i, value in enumerate(iterable) if value == target]


def copy_and_append(seq: Iterable[T], item: T) -> list[T]:
    out = list(seq)
    out.append(item)
    return out
