from collections.abc import Sequence
from enum import Enum

import numpy
import numpy as np
from numpy import ndarray


class PerformanceMeasure(Enum):
    balanced_accuracy = 1
    precision = 2
    recall = 3
    specificity = 4


class ConfusionMatrix:
    __matrix: ndarray
    __class_labels: Sequence[str]

    def __init__(self, matrix: ndarray, class_labels: Sequence[str]):
        self.__matrix = matrix
        self.__class_labels = class_labels

    @classmethod
    def create_from_seq(cls, seq: Sequence[int], labels: Sequence[str]):
        matrix = numpy.array(seq)
        n_classes = len(labels)
        matrix = matrix.reshape((n_classes, n_classes))
        return ConfusionMatrix(matrix=matrix, class_labels=labels)

    def n_samples(self) -> int:
        return self.__matrix.sum()

    def samples_by_true_class(self) -> ndarray:
        return self.__matrix.sum(axis=0)

    def samples_by_prediction(self) -> ndarray:
        return self.__matrix.sum(axis=1)

    def tp(self) -> ndarray:
        return np.diag(self.__matrix)

    def fp(self) -> ndarray:
        return self.samples_by_true_class() - self.tp()

    def fn(self) -> ndarray:
        return self.samples_by_prediction() - self.tp()

    def tn(self) -> ndarray:
        return self.n_samples() - (self.fp() + self.fn() + self.tp())

    def recall(self) -> ndarray:
        """Recall, sensitivity, hit rate, or true positive rate."""
        tp = self.tp()
        with np.errstate(divide='ignore', invalid='ignore'):
            #  Ignoring division by zero that can legitimately happen.
            return tp / (tp + self.fn())

    def specificity(self) -> ndarray:
        """Specificity, selectivity or true negative rate"""
        tn = self.tn()
        with np.errstate(divide='ignore', invalid='ignore'):
            #  Ignoring division by zero that can legitimately happen.
            return tn / (tn + self.fp())

    def precision(self) -> ndarray:
        """Precision or positive predictive value."""
        tp = self.tp()
        with np.errstate(divide='ignore', invalid='ignore'):
            #  Ignoring division by zero that can legitimately happen.
            return tp / (tp + self.fp())

    def n_correct_predictions(self) -> int:
        return sum(self.tp())

    def balanced_accuracies(self) -> ndarray:
        return (self.recall() + self.specificity()) / 2.0

    def labels(self) -> Sequence[str]:
        return self.__class_labels

    def performance_measure(self, measure: PerformanceMeasure) -> ndarray:
        if measure == PerformanceMeasure.balanced_accuracy:
            return self.balanced_accuracies()
        elif measure == PerformanceMeasure.precision:
            return self.precision()
        elif measure == PerformanceMeasure.recall:
            return self.recall()
        elif measure == PerformanceMeasure.specificity:
            return self.specificity()
        else:
            raise ValueError()
