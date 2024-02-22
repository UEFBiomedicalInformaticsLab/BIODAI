from __future__ import annotations
from abc import ABC
from collections.abc import Iterable
from copy import deepcopy

from individual.Individual import Individual
from individual.fit import Fit
from individual.fitness.high_best_fitness import HighBestFitness


class FitIndividual(Individual, Fit, ABC):
    fitness: HighBestFitness  # Fitness is not private to keep some backward compatibility with Deap.

    def __init__(self, fitness: HighBestFitness):
        self.fitness = fitness

    def has_fitness(self):
        return True

    def get_objective_fitness(self, fitness_index):
        return self.fitness.values[fitness_index]

    def get_test_fitness(self) -> HighBestFitness:
        return deepcopy(self.fitness)

    def n_objectives(self) -> int:
        return self.fitness.n_objectives()



def get_fitnesses(pop: Iterable[FitIndividual], fitness_index) -> list[float]:
    res = []
    for i in pop:
        res.append(i.get_objective_fitness(fitness_index))
    return res
