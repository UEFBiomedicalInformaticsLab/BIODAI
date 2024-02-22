from random import random


def mut_flip_bit(individual, indpb, mask=None):
    """ As DEAP mutFlipBit but accepts a mask of active bits.
        Flip the value of the attributes of the input individual and return the
        mutant. The *individual* is expected to be a :term:`sequence` and the values of the
        attributes shall stay valid after the ``not`` operator is called on them.
        The *indpb* argument is the probability of each attribute to be
        flipped. This mutation is usually applied on boolean individuals.

        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be flipped.
        :param mask: bits that are allowed to flip.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
    if mask is None:
        for i in range(len(individual)):
            if random() < indpb:
                individual[i] = type(individual[i])(not individual[i])
    else:
        for i in range(len(individual)):
            if mask[i]:
                if random() < indpb:
                    individual[i] = type(individual[i])(not individual[i])

    return individual,
