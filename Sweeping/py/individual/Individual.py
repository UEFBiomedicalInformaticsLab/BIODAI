from abc import ABC

from util.list_like import ListLike


class Individual(ListLike, ABC):
    """The length of an individual is the number of hyperparameters that can be selected,
    according to the representation of this kind of individual.
    It cannot be assumed to be always the number of predictive features.
    In order to get the predictive or used features of an individual an
    appropriate hyperparameter manager must be used.
    Equality of two individuals makes sense only if they share the same representation of
    the hyperparameters.
    Hash is not implemented because of mutability."""

    def has_fitness(self):
        return False
