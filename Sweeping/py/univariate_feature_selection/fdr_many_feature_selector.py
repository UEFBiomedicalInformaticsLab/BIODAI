from collections.abc import Sequence
from typing import Optional

import numpy as np

from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from univariate_feature_selection.many_feature_selector import ManyFeatureSelector
from univariate_feature_selection.univariate_feature_selector_descriptor import DEFAULT_FDR_THRESHOLD, FDR_STR, \
    FdrManyFeatureSelectorDescriptor
from univariate_property_computer.parallel_univariate_property_computer import compute_univariate_property_with_workers
from univariate_property_computer.univariate_pval_computer import UnivariatePvalComputer, LogUnivariatePvalComputer, \
    AnovaUnivariatePvalComputer
from util.printer.printer import Printer, DEFAULT_PRINTER
from util.table.table import Table


def select_with_fdr_from_pvals(p_values: Sequence[float], fdr_threshold: float = DEFAULT_FDR_THRESHOLD) -> list[bool]:
    p_values = np.array(p_values)
    num_tests = len(p_values)

    # Identify valid (non-NaN) p-values
    valid_mask = ~np.isnan(p_values)
    valid_p_values = p_values[valid_mask]
    num_valid = len(valid_p_values)

    if num_valid == 0:
        return [False] * num_tests  # All NaNs, nothing to select

    # Sort valid p-values and get their ranks
    sorted_indices = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sorted_indices]
    ranks = np.arange(1, num_valid + 1)
    thresholds = ranks / num_valid * fdr_threshold

    # Determine significance
    significant = sorted_p_values <= thresholds
    significant_indices = np.where(significant)[0]

    selected_features_bool = np.zeros(num_tests, dtype=bool)

    if significant_indices.size > 0:
        max_significant_index = significant_indices.max()
        selected_valid_indices = sorted_indices[:max_significant_index + 1]
        selected_features_bool[np.where(valid_mask)[0][selected_valid_indices]] = True

    return selected_features_bool.tolist()


class FdrManyFeatureSelector(ManyFeatureSelector):
    __computer: UnivariatePvalComputer

    def __init__(self, computer: UnivariatePvalComputer,
                 fdr_threshold: float = DEFAULT_FDR_THRESHOLD):
        ManyFeatureSelector.__init__(
            self=self,
            descriptor=FdrManyFeatureSelectorDescriptor(
                computer=computer.descriptor(),
                fdr_threshold=fdr_threshold))
        self.__computer = computer

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        pvals = compute_univariate_property_with_workers(
            single_feature_computer=self.__computer, data=data, outcome=outcome,
            n_proc=n_proc, task_name="Computing p-values", covariates=covariates, printer=printer)
        n_pvals = len(pvals)
        if n_pvals > 0:
            try:
                tot_nan = np.isnan(pvals).sum()
                printer.print("" + str(tot_nan / len(pvals)) + " prevalence of NaNs in p-values.")
            except TypeError as e:
                printer.print(
                    "It is not possible to compute the prevalence of NaNs in p-values because of a TypeError.")
                printer.print(str(e) + "\n" + "pvals type: " + str(type(pvals)) + "\n" + "pvals:\n" + str(pvals))
        else:
            printer.print("The function returned an empty sequence of p-values.")
        return select_with_fdr_from_pvals(p_values=pvals, fdr_threshold=self.fdr_threshold())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return self.__computer.outcome_types()

    def __str__(self) -> str:
        return ("FDR many feature selector with p-val computer " +
                str(self.__computer) + " and FDR threshold " + self._fdr_str())

    def fdr_threshold(self) -> float:
        return self.descriptor().fdr_threshold()

    def _computer(self) -> UnivariatePvalComputer:
        return self.__computer

    def descriptor(self) -> FdrManyFeatureSelectorDescriptor:
        res = ManyFeatureSelector.descriptor(self=self)
        assert isinstance(res, FdrManyFeatureSelectorDescriptor)
        return res

    def _fdr_str(self) -> str:
        return self.descriptor().fdr_str()

    def ignores_covariates(self) -> bool:
        return self.__computer.ignores_covariates()

    def name(self) -> str:
        return self.__computer.name() + " " + FDR_STR + " " + self._fdr_str()

    def main_nick(self) -> str:
        return self.descriptor().algorithm_nick()


class LogisticFdrSelector(FdrManyFeatureSelector):

    def __init__(self, fdr_threshold: float = DEFAULT_FDR_THRESHOLD):
        FdrManyFeatureSelector.__init__(self=self, computer=LogUnivariatePvalComputer(), fdr_threshold=fdr_threshold)


class AnovaFdrSelector(FdrManyFeatureSelector):

    def __init__(self, fdr_threshold: float = DEFAULT_FDR_THRESHOLD):
        FdrManyFeatureSelector.__init__(self=self, computer=AnovaUnivariatePvalComputer(), fdr_threshold=fdr_threshold)
