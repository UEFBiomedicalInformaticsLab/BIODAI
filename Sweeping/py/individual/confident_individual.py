from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Optional, Iterable

from individual.fit_individual import FitIndividual
from individual.fitness.high_best_fitness import HighBestFitness
from util.hyperbox.hyperbox import Interval


class ConfidentIndividual(FitIndividual, ABC):
    __std_dev: Sequence[Optional[float]]
    __ci95: Sequence[Optional[Interval]]
    __bootstrap_mean: Sequence[Optional[float]]

    def __init__(self, fitness: HighBestFitness):
        FitIndividual.__init__(self, fitness=fitness)
        n_objectives =fitness.n_objectives()
        self.__std_dev = [None] * n_objectives
        self.__ci95 = [None] * n_objectives
        self.__bootstrap_mean = [None] * n_objectives

    def set_std_dev(self, std_dev: Sequence[Optional[float]]):
        if len(std_dev) != self.n_objectives():
            raise ValueError(
                "Passed standard deviations are in a wrong number.\n" +
                "Passed standard deviations: " + str(std_dev) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__std_dev = std_dev

    def set_ci95(self, ci95: Sequence[Optional[Interval]]):
        if len(ci95) != self.n_objectives():
            raise ValueError(
                "Passed 95% confidence intervals are in a wrong number.\n" +
                "Passed intervals: " + str(ci95) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__ci95 = ci95

    def std_dev(self) -> Sequence[Optional[float]]:
        return self.__std_dev

    def ci95(self) -> Sequence[Optional[Interval]]:
        return self.__ci95

    def set_bootstrap_mean(self, bootstrap_mean: Sequence[Optional[float]]):
        if len(bootstrap_mean) != self.n_objectives():
            raise ValueError(
                "Passed bootstrap means are in a wrong number.\n" +
                "Passed bootstrap means: " + str(bootstrap_mean) + "\n" +
                "Individual: " + str(self) + "\n")
        self.__bootstrap_mean = bootstrap_mean

    def bootstrap_mean(self) -> Sequence[Optional[float]]:
        return self.__bootstrap_mean


def get_ci95s(pop: Iterable[ConfidentIndividual], fitness_index) -> list[Interval]:
    res = []
    for i in pop:
        res.append(i.ci95()[fitness_index])
    return res


def get_std_devs(pop: Iterable[ConfidentIndividual], fitness_index) -> list[float]:
    res = []
    for i in pop:
        res.append(i.std_dev()[fitness_index])
    return res


def get_bootstrap_means(pop: Iterable[ConfidentIndividual], fitness_index) -> list[float]:
    res = []
    for i in pop:
        res.append(i.bootstrap_mean()[fitness_index])
    return res
