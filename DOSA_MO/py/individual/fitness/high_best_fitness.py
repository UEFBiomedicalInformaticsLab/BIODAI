from collections.abc import Sequence

from individual.fitness.fitness import Fitness


class HighBestFitness(Fitness):
    """Should implement base.Fitness according to DEAP, but base.Fitness has the abstract weights class attribute
    that would require to extend with a different class for each possible number of dimensions."""

    def __init__(self, n_objectives: int, values: Sequence[float] = ()):
        weights = tuple(1.0 for _ in range(n_objectives))
        Fitness.__init__(self=self, weights=weights, values=values)
