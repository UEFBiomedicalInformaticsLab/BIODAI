from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Iterable

from individual.fit_individual import FitIndividual
from util.hyperbox.hyperbox import Interval


class ConfidentIndividual(FitIndividual, ABC):

    @abstractmethod
    def std_dev(self) -> Sequence[Optional[float]]:
        raise NotImplementedError()

    @abstractmethod
    def ci95(self) -> Sequence[Optional[Interval]]:
        raise NotImplementedError()

    @abstractmethod
    def bootstrap_mean(self):
        raise NotImplementedError()


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
