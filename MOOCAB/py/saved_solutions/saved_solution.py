from typing import Optional, Sequence

from individual.fit import Fit
from individual.fitness.high_best_fitness import HighBestFitness
from input_data.view_prefix import view_number
from objective.objective_with_importance.leanness import SoftLeanness
from prediction_stats.confusion_matrix import ConfusionMatrix
from util.hyperbox.hyperbox import ConcreteHyperbox0B, Interval
from util.math.list_math import list_div, list_add, list_subtract
from util.utils import IllegalStateError


class SavedSolution(Fit):
    __features: Optional[Sequence[str]]
    __confusion_matrix: Optional[ConfusionMatrix]
    __train_fitnesses: Optional[Sequence[float]]
    # Fitnesses evaluated during optimization, like with inner cross-validation.
    __test_fitnesses: Optional[Sequence[float]]
    # Fitnesses evaluated after optimization on new data.
    __train_std_devs: Optional[Sequence[float]]
    __test_std_devs: Optional[Sequence[float]]
    __train_ci: Optional[Sequence[Interval]]
    __test_ci: Optional[Sequence[Interval]]

    def __init__(self,
                 train_fitnesses: Optional[Sequence[float]],
                 test_fitnesses: Optional[Sequence[float]],
                 features: Optional[Sequence[str]],
                 confusion_matrix: Optional[ConfusionMatrix],
                 train_std_devs: Optional[Sequence[float]] = None,
                 test_std_devs: Optional[Sequence[float]] = None,
                 train_ci: Optional[Sequence[Interval]] = None,
                 test_ci: Optional[Sequence[Interval]] = None):
        self.__train_fitnesses = train_fitnesses
        self.__test_fitnesses = test_fitnesses
        self.__features = features
        self.__confusion_matrix = confusion_matrix
        self.__train_std_devs = train_std_devs
        self.__test_std_devs = test_std_devs
        self.__train_ci = train_ci
        self.__test_ci = test_ci

    def has_test_fitnesses(self) -> bool:
        return self.__test_fitnesses is not None

    def test_fitnesses(self) -> Sequence[float]:
        if self.has_test_fitnesses():
            return self.__test_fitnesses
        else:
            raise IllegalStateError()

    def num_fitnesses(self) -> int:
        if self.has_test_fitnesses():
            return len(self.__test_fitnesses)
        else:
            raise IllegalStateError()

    def has_train_fitnesses(self) -> bool:
        """Fitnesses evaluated during optimization, like with inner cross-validation."""
        return self.__train_fitnesses is not None

    def train_fitnesses(self) -> Sequence[float]:
        if self.has_train_fitnesses():
            return self.__train_fitnesses
        else:
            raise IllegalStateError()

    def train_hyperbox(self) -> ConcreteHyperbox0B:
        """Hyperbox for fitnesses evaluated during optimization, like with inner cross-validation."""
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

    def num_features_by_view(self) -> dict[int, int]:
        feat = self.features()
        res_dict = {}
        for f in feat:
            view_index = view_number(feature_name=f)
            if view_index in res_dict:
                res_dict[view_index] += 1
            else:
                res_dict[view_index] = 1
        return res_dict

    def view_prevalences(self) -> dict[int, float]:
        counts = self.num_features_by_view()
        tot = sum(counts.values())
        res = {}
        for k in counts:
            if tot == 0:
                res[k] = 0.0
            else:
                res[k] = float(counts[k])/float(tot)
        return res

    def get_test_fitness(self) -> HighBestFitness:
        fit_tup = tuple(self.test_fitnesses())
        return HighBestFitness(n_objectives=len(fit_tup), values=fit_tup)

    def has_confusion_matrix(self) -> bool:
        return self.__confusion_matrix is not None

    def confusion_matrix(self) -> ConfusionMatrix:
        if self.has_confusion_matrix():
            return self.__confusion_matrix
        else:
            raise IllegalStateError()

    def has_train_std_devs(self) -> bool:
        return self.__train_std_devs is not None

    def has_test_std_devs(self) -> bool:
        return self.__test_std_devs is not None

    def train_std_devs(self) -> Sequence[float]:
        if self.has_train_std_devs():
            return self.__train_std_devs
        else:
            raise IllegalStateError()

    def test_std_devs(self) -> Sequence[float]:
        if self.has_test_std_devs():
            return self.__test_std_devs
        else:
            raise IllegalStateError()

    def has_train_ci(self) -> bool:
        return self.__train_ci is not None

    def has_test_ci(self) -> bool:
        return self.__test_ci is not None

    def train_ci(self) -> Sequence[Interval]:
        if self.has_train_ci():
            return self.__train_ci
        else:
            raise IllegalStateError()

    def test_ci(self) -> Sequence[Interval]:
        if self.has_test_ci():
            return self.__test_ci
        else:
            raise IllegalStateError()

    def has_performance_gap(self) -> bool:
        return self.has_train_fitnesses() and self.has_test_fitnesses()

    def performance_gap(self) -> Sequence[float]:
        if self.has_performance_gap():
            return list_subtract(self.train_fitnesses(), self.test_fitnesses())
        else:
            raise IllegalStateError()

    def __str__(self) -> str:
        res = ""
        if self.has_train_fitnesses():
            res += "Train fitnesses: " + str(self.train_fitnesses()) + "\n"
        if self.has_train_std_devs():
            res += "Train standard deviations: " + str(self.train_std_devs()) + "\n"
        if self.has_train_ci():
            res += "Train confidence interval 95%: " + str(self.train_ci()) + "\n"
        if self.has_test_fitnesses():
            res += "Test fitnesses: " + str(self.test_fitnesses()) + "\n"
        if self.has_test_std_devs():
            res += "Test standard deviations: " + str(self.test_std_devs()) + "\n"
        if self.has_test_ci():
            res += "Test confidence interval 95%: " + str(self.test_ci()) + "\n"
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
