from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

from util.sequence_utils import tuple_to_string


class Hyperbox(ABC):
    """A hyperbox can have zero or more dimensions. Borders are included in the hyperbox."""

    @abstractmethod
    def intervals(self) -> Sequence[Interval]:
        raise NotImplementedError()

    def n_dimensions(self) -> int:
        return len(self.intervals())

    def intersects_hyperbox(self, other: Hyperbox) -> bool:
        """If the number of dimensions differ result is unspecified.
        Zero-dimensional hyperboxes always intersect."""
        for s, o in zip(self.intervals(), other.intervals()):
            if not s.intersects_interval(o):
                return False
        return True

    def intersects_hyperbox0b(self, other: Hyperbox0B) -> bool:
        """If the number of dimensions differ result is unspecified.
        Zero-dimensional hyperboxes always intersect."""
        for s, o in zip(self.intervals(), other.intervals()):
            if not s.intersects_interval0b(o):
                return False
        return True

    def intersection_hyperbox(self, other: Hyperbox) -> Hyperbox:
        res_intervals = []
        for s, o in zip(self.intervals(), other.intervals()):
            res_intervals.append(s.intersection_interval(o))
        return ConcreteHyperbox(intervals=res_intervals)

    def intersection_hyperbox0b(self, other: Hyperbox0B) -> Hyperbox:
        res_intervals = []
        for s, o in zip(self.intervals(), other.intervals()):
            res_intervals.append(s.intersection_interval0b(o))
        return ConcreteHyperbox(intervals=res_intervals)

    def volume(self) -> float:
        res = 1.0
        for i in self.intervals():
            res = res * i.length()
        return res

    def project(self, d: int) -> Hyperbox:
        """Projection of the hyperbox on the hyperplane orthogonal to the d-th dimension. In practice the
        d-th dimension is removed."""
        res_intervals = []
        intervals = self.intervals()
        for i in range(self.n_dimensions()):
            if i is not d:
                res_intervals.append(intervals[i])
        return ConcreteHyperbox(intervals=res_intervals)

    def contains_point(self, point: Sequence[float]) -> bool:
        """If the number of dimensions differ result is unspecified."""
        for i, c in zip(self.intervals(), point):
            if not i.contains_pos(c):
                return False
        return True

    def __str__(self) -> str:
        return tuple_to_string(self.intervals())


class ConcreteHyperbox(Hyperbox):
    __intervals: Sequence[Interval]

    def __init__(self, intervals: Sequence[Interval]):
        self.__intervals = intervals

    def intervals(self) -> Sequence[Interval]:
        return self.__intervals


class Hyperbox0B(Hyperbox, ABC):

    @abstractmethod
    def intervals(self) -> Sequence[Interval0B]:
        raise NotImplementedError()

    def intersects_hyperbox0b(self, other: Hyperbox0B) -> bool:
        return True

    def intersection_hyperbox0b(self, other: Hyperbox0B) -> Hyperbox0B:
        res_intervals = []
        for s, o in zip(self.intervals(), other.intervals()):
            res_intervals.append(s.intersection_interval0b(o))
        return ConcreteHyperbox0B(intervals=res_intervals)

    def project(self, d: int) -> Hyperbox0B:
        """Projection of the hyperbox on the hyperplane orthogonal to the d-th dimension. In practice the
        d-th dimension is removed."""
        res_intervals = []
        intervals = self.intervals()
        for i in range(self.n_dimensions()):
            if i is not d:
                res_intervals.append(intervals[i])
        return ConcreteHyperbox0B(intervals=res_intervals)


class ConcreteHyperbox0B(Hyperbox0B):
    __intervals: Sequence[Interval0B]

    def __init__(self, intervals: Sequence[Interval0B]):
        self.__intervals = intervals

    def intervals(self) -> Sequence[Interval0B]:
        return self.__intervals

    @staticmethod
    def create_by_b_vals(b_vals: Sequence[float]) -> ConcreteHyperbox0B:
        intervals = [Interval0B(b) for b in b_vals]
        return ConcreteHyperbox0B(intervals=intervals)


class Interval(Hyperbox, ABC):
    """An interval is a one dimensional hyperbox."""

    @abstractmethod
    def a(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def b(self) -> float:
        raise NotImplementedError()

    def n_dimensions(self):
        return 1

    def intervals(self) -> Sequence[Interval]:
        return [self]

    def intersects_interval(self, other: Interval) -> bool:
        return self.a() <= other.b() and other.a() <= self.b()

    def intersection_interval(self, other: Interval) -> Interval:
        res_a = max(self.a(), other.a())
        res_b = max(self.b(), other.b())
        return ConcreteInterval(a=res_a, b=res_b)  # You get an error if there is no intersection.

    def intersects_interval0b(self, other: Interval0B) -> bool:
        return self.a() <= other.b() and 0.0 <= self.b()

    def intersection_interval0b(self, other: Interval0B) -> Interval:
        res_a = max(self.a(), 0.0)
        res_b = max(self.b(), other.b())
        return ConcreteInterval(a=res_a, b=res_b)  # You get an error if there is no intersection.

    def length(self) -> float:
        return self.b() - self.a()

    def contains_pos(self, pos: float) -> bool:
        return self.a() <= pos <= self.b()

    def mid_pos(self) -> float:
        return (self.b() + self.a()) / 2.0

    def __str__(self) -> str:
        return "[" + str(self.a()) + ", " + str(self.b()) + "]"


class Interval0B(Interval):
    __b: float

    def __init__(self, b: float):
        if b < 0.0:
            raise ValueError()
        self.__b = b

    def a(self) -> float:
        return 0.0

    def b(self) -> float:
        return self.__b

    def intersects_interval0b(self, other: Interval0B) -> bool:
        return True

    def intersection_interval0b(self, other: Interval0B) -> Interval0B:
        return Interval0B(min(self.__b, other.__b))

    def intersects_interval(self, other: Interval) -> bool:
        return 0.0 <= other.b() and other.a() <= self.__b

    def intersection_interval(self, other: Interval) -> Interval:
        res_b = max(self.__b, other.b())
        o_a = other.a()
        if o_a > 0.0:
            return ConcreteInterval(a=o_a, b=res_b)  # You get an error if there is no intersection.
        else:
            return Interval0B(b=res_b)  # You get an error if there is no intersection.

    def length(self) -> float:
        return self.b()

    def contains_pos(self, pos: float) -> bool:
        return 0.0 <= pos <= self.__b

    def mid_pos(self) -> float:
        return self.b() / 2.0


class ConcreteInterval(Interval):
    __a: float
    __b: float

    def __init__(self, a: float, b: float):
        if a > b:
            raise ValueError()
        self.__a = a
        self.__b = b

    def a(self) -> float:
        return self.__a

    def b(self) -> float:
        return self.__b
