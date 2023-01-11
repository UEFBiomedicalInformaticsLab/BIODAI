from __future__ import annotations
import sys
from collections import Sequence
from operator import mul, truediv

from util.utils import IllegalStateError


class Fitness(object):
    """Based on DEAP base.Fitness. base.Fitness has the abstract weights class attribute
    that would require to extend with a different class for each possible number of dimensions. Here we
    substitute it with an instance attribute of the same name.

    The fitness is a measure of quality of a solution. If *values* are
    provided as a tuple, the fitness is initialized using those values,
    otherwise it is empty (or invalid).

    :param values: The initial values of the fitness as a tuple, optional.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made lexicographically.
    Maximization and minimization are taken care off by a multiplication
    between the :attr:`weights` and the fitness :attr:`values`. The comparison
    can be made between fitnesses of different size, if the fitnesses are
    equal until the extra elements, the longer fitness will be superior to the
    shorter.

    Different types of fitnesses are created in the :ref:`creating-types`
    tutorial.

    .. note::
       When comparing fitness values that are **minimized**, ``a > b`` will
       return :data:`True` if *a* is **smaller** than *b*.
    """

    weights: Sequence[float]
    """# According to DEAP framework this should be a class attribute. We want it as instance
    # attribute in order to set the number of objectives at runtime.
    The weights are used in the fitness comparison. The
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization."""

    wvalues = Sequence[float]
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.

    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """

    def __init__(self, weights: Sequence[float], values=()):
        self.weights = weights

        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."
                            % self.__class__)

        if len(values) > 0:
            self.values = values
            self.wvalues = tuple(map(mul, values, self.weights))
        else:
            self.wvalues = ()

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        assert len(values) == len(self.weights), "Assigned values have not the same length than fitness weights"
        for v in values:
            if not isinstance(v, float):
                raise ValueError("Passed values: " + str(values))
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
                              "sequence of numbers when assigning to values of "
                              "%r. Currently assigning value(s) %r of %r to a "
                              "fitness with weights %s."
                              % (self.__class__, values, type(values),
                                 self.weights)).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self) -> int:
        try:
            return hash(self.wvalues)
        except BaseException as e:
            raise IllegalStateError("Exception while hasing Fitness.\n +"
                                    "values: " + str(self.values) + "\n" +
                                    "wvalues: " + str(self.wvalues) + "\n" +
                                    "original exception: " + str(e) + "\n")

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())

    def n_objectives(self) -> int:
        return len(self.weights)

    def as_list(self) -> list[float]:
        if self.valid:
            return list(self.values)
        else:
            return []
