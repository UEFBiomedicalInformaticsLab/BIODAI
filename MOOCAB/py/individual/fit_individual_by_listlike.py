from abc import ABC

from individual.fit_individual import FitIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from individual.individual_by_listlike import IndividualByListlike


class FitIndividualByListlike(FitIndividual, IndividualByListlike, ABC):

    def __init__(self, fitness: HighBestFitness, seq=()):
        FitIndividual.__init__(self, fitness=fitness)
        IndividualByListlike.__init__(self, seq=seq)

