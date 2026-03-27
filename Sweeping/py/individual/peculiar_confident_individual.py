from abc import ABC

from individual.confident_individual import ConfidentIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from individual.peculiar_individual import PeculiarIndividual
from individual.predictive_individual import PredictiveIndividualSV, PredictiveIndividualMV


class PeculiarConfidentIndividual(PeculiarIndividual, ConfidentIndividual, ABC):

    def __init__(self, n_objectives: int):
        PeculiarIndividual.__init__(self=self, n_objectives=n_objectives)


class PeculiarConfidentIndividualSV(PeculiarConfidentIndividual, PredictiveIndividualSV, ABC):

    def __init__(self, fitness: HighBestFitness):
        PeculiarConfidentIndividual.__init__(self=self, n_objectives=fitness.n_objectives())
        PredictiveIndividualSV.__init__(self=self, fitness=fitness)


class PeculiarConfidentIndividualMV(PeculiarConfidentIndividual, PredictiveIndividualMV, ABC):

    def __init__(self, fitness: HighBestFitness):
        PeculiarConfidentIndividual.__init__(self=self, n_objectives=fitness.n_objectives())
        PredictiveIndividualMV.__init__(self=self, fitness=fitness)