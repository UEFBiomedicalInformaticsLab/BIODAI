from typing import Optional, Sequence

from individual.fit import Fit
from individual.fitness.high_best_fitness import HighBestFitness
from objective.objective_computer import SoftLeanness
from prediction_stats.confusion_matrix import ConfusionMatrix
from util.hyperbox.hyperbox import ConcreteHyperbox0B
from util.list_math import list_div, list_add
from util.utils import IllegalStateError


class SavedSolution(Fit):
    __features: Optional[Sequence[str]]
    __confusion_matrix: Optional[ConfusionMatrix]
    __train_fitnesses: Optional[Sequence[float]]
    __test_fitnesses: Optional[Sequence[float]]

    def __init__(self,
                 train_fitnesses: Optional[Sequence[float]],
                 test_fitnesses: Optional[Sequence[float]],
                 features: Optional[Sequence[str]],
                 confusion_matrix: Optional[ConfusionMatrix]):
        self.__train_fitnesses = train_fitnesses
        self.__test_fitnesses = test_fitnesses
        self.__features = features
        self.__confusion_matrix = confusion_matrix

    def has_fitnesses(self) -> bool:
        return self.__test_fitnesses is not None

    def test_fitness(self) -> Sequence[float]:
        if self.has_fitnesses():
            return self.__test_fitnesses
        else:
            raise IllegalStateError()

    def train_hyperbox(self) -> ConcreteHyperbox0B:
        return ConcreteHyperbox0B.create_by_b_vals(self.__train_fitnesses)

    def test_hyperbox(self) -> ConcreteHyperbox0B:
        return ConcreteHyperbox0B.create_by_b_vals(self.__test_fitnesses)

    def train_soft_hyperbox(self) -> ConcreteHyperbox0B:
        """Converts fitness in position 1 from leanness to soft-leanness"""
        fit = list(self.__train_fitnesses)
        fit[1] = SoftLeanness.leanness_to_soft_leanness(fit[1])
        return ConcreteHyperbox0B.create_by_b_vals(fit)

    def test_soft_hyperbox(self) -> ConcreteHyperbox0B:
        """Converts fitness in position 1 from leanness to soft-leanness"""
        fit = list(self.__train_fitnesses)
        fit[1] = SoftLeanness.leanness_to_soft_leanness(fit[1])
        return ConcreteHyperbox0B.create_by_b_vals(fit)

    def has_features(self) -> bool:
        return self.__features is not None

    def num_features(self) -> int:
        return len(self.features())

    def features(self) -> Sequence[str]:
        if self.has_features():
            return self.__features
        else:
            raise IllegalStateError()

    def get_test_fitness(self) -> HighBestFitness:
        fit_tup = tuple(self.test_fitness())
        return HighBestFitness(n_objectives=len(fit_tup), values=fit_tup)

    def has_confusion_matrix(self) -> bool:
        return self.__confusion_matrix is not None

    def confusion_matrix(self) -> ConfusionMatrix:
        if self.has_confusion_matrix():
            return self.__confusion_matrix
        else:
            raise IllegalStateError()

    def __str__(self) -> str:
        res = ""
        if self.__train_fitnesses is not None:
            res += str(self.__train_fitnesses)
        if self.__test_fitnesses is not None:
            res += str(self.__test_fitnesses)
        if self.__features is not None:
            res += str(self.__features)
        return res


def union_of_features(solutions: Sequence[SavedSolution]) -> Sequence[str]:
    res = set()
    for s in solutions:
        if s.has_features():
            sf = s.features()
            res = res.union(sf)
    return list(res)


def feature_list_to_mask(feature_list: Sequence[str], all_features: Sequence[str]) -> list[bool]:
    return [f in feature_list for f in all_features]


def average_individual(fold_solutions: Sequence[SavedSolution], all_features: Sequence[str]) -> list[float]:
    """The average individual is an individual with for each gene the frequency in the population."""
    tot_features = sum_of_individuals(fold_solutions=fold_solutions, all_features=all_features)
    return list_div(tot_features, len(fold_solutions))


def sum_of_individuals(fold_solutions: Sequence[SavedSolution], all_features: Sequence[str]) -> list[int]:
    tot_features = [0]*len(all_features)
    for f in fold_solutions:
        if f.has_features():
            tot_features = list_add(
                tot_features, feature_list_to_mask(feature_list=f.features(), all_features=all_features))
    return tot_features
