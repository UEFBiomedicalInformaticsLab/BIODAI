from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Union

from numpy import ravel
from pandas import DataFrame

from cross_validation.single_objective.cv_result import CVResult
from hyperparam_manager.hyperparam_manager import HyperparamManager
from objective.objective_with_importance.structural_objective_computer_with_importance import \
    StructuralObjectiveComputerWithImportance
from math import sqrt
import math
from sklearn.utils.multiclass import unique_labels

from util.dataframes import n_row, n_col
from util.math.online_variance_builder import OnlineVarianceBuilder
from util.math.variance_builder import VarianceBuilder


class Separation(StructuralObjectiveComputerWithImportance, ABC):

    @staticmethod
    def requires_target() -> bool:
        return True

    @staticmethod
    def __separation(mean1: float, mean2: float, var1: float, var2: float) -> float:
        diff = abs(mean1 - mean2)
        den = sqrt(var1 + var2)
        if den == 0.0:
            if diff == 0.0:
                return 0.0
            else:
                return math.inf
        else:
            return diff / den

    @staticmethod
    def __separation_from_builders(builder1: VarianceBuilder, builder2: VarianceBuilder) -> float:
        if builder1.has_mean() and builder2.has_mean():
            if builder1.has_variance():
                var1 = builder1.unbiased_variance()
            else:
                var1 = 0.0
            if builder2.has_variance():
                var2 = builder2.unbiased_variance()
            else:
                var2 = 0.0
            return Separation.__separation(
                mean1=builder1.mean(),
                mean2=builder2.mean(),
                var1=var1,
                var2=var2)
        else:
            return math.inf

    @staticmethod
    def __classes_separation(
            builders: dict[tuple[str, str], VarianceBuilder], columns: Sequence[str], label1, label2) -> float:
        separation = 0.0
        for c in columns:
            builder1 = builders[c, label1]
            builder2 = builders[c, label2]
            separation = max(
                separation,
                Separation.__separation_from_builders(
                    builder1=builder1,
                    builder2=builder2
                ))
        return separation

    @staticmethod
    def __create_builders(masked_x: DataFrame, y: DataFrame) -> dict[tuple[str, str], VarianceBuilder]:
        y = ravel(y)
        n_samples = len(y)
        if n_row(masked_x) != n_samples:
            raise ValueError("number of rows: " + str(n_row(masked_x)) + "\n" +
                             "length of y: " + str(n_samples) + "\n")
        labels = unique_labels(y)
        label_masks = {}
        for lab in labels:
            label_masks[lab] = [y[i] == lab for i in range(n_samples)]
        builders = {}
        for c in masked_x.columns:
            col = masked_x[c]
            for la in labels:
                builder = OnlineVarianceBuilder()
                builder.add_all(col.iloc[label_masks[la]])
                builders[(c, la)] = builder
        return builders

    @staticmethod
    def __create_builders_old(masked_x: DataFrame, y: DataFrame) -> dict[tuple[str, str], VarianceBuilder]:
        """Old slower method."""
        y = ravel(y)
        n_samples = len(y)
        labels = unique_labels(y)
        builders = {}
        for c in masked_x.columns:
            col = masked_x[c]
            if len(col) != n_samples:
                raise ValueError("length of the column: " + str(len(col)) + "\n" +
                                 "length of y: " + str(n_samples) + "\n")
            for la in labels:
                builders[(c, la)] = OnlineVarianceBuilder()
            for i in range(n_samples):
                builders[(c, y[i])].add(col.iloc[i])
        return builders

    @abstractmethod
    def _separation_from_class_separations(self, class_separations: Sequence[float]) -> float:
        raise NotImplementedError()

    def compute_from_structure(self, hyperparams, hp_manager: Union[HyperparamManager, None],
                               x: DataFrame, y: DataFrame) -> CVResult:
        if n_col(x) != hp_manager.n_active_features(hyperparams):
            raise ValueError("x cols: " + str(n_col(x)) +
                             "\nActive features from individual: " + str(hp_manager.n_active_features(hyperparams)))
        labels = unique_labels(y)
        n_labels = len(labels)
        builders = Separation.__create_builders(masked_x=x, y=y)
        class_separations = []
        for i1 in range(n_labels):
            for i2 in range(i1+1, n_labels):
                class_separations.append(self.__classes_separation(
                    builders=builders, columns=x.columns, label1=labels[i1], label2=labels[i2]))
        separation = self._separation_from_class_separations(class_separations=class_separations)
        if separation == math.inf:
            return CVResult(fitness=1.0)
        else:
            return CVResult(fitness=separation / (1.0 + separation))
