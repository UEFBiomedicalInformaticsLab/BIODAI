import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Optional, Any

import pandas as pd
from numpy import number
from pandas import DataFrame

from descriptor.descriptor import Described
from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType, ALL_OUTCOME_TYPES
from model.survival.survival_model import LifelinesModel
from univariate_feature_selection.parallel_anova import anova_filter_one_feature_checked
from consts import DEFAULT_P_VAL
from univariate_feature_selection.univariate_feature_selector_descriptor import SingleFeatureSelectorDescriptor, \
    DummySingleFeatureSelectorDescriptor, DUMMY_SELECTOR_SINGLE_DESCRIPTOR, DEFAULT_MISSING_THRESHOLD, \
    DEFAULT_MINOR_FREQUENCY_THRESHOLD, DEFAULT_HWE_PVAL, MissingSingleFeatureSelectorDescriptor, \
    MinorFrequencySingleFeatureSelectorDescriptor, WithPvalDescriptor, SingleFeatureSelectorAnovaCategoricalDescriptor, \
    SingleFeatureSelectorAnovaSurvivalDescriptor, SingleFeatureSelectorCoxDescriptor, \
    CompositeSingleFeatureSelectorDescriptor, HWESingleFeatureSelectorDescriptor, DEFAULT_MAF_THRESHOLD, \
    MAFSingleFeatureSelectorDescriptor
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from util.survival.survival_utils import survival_events
from util.table.table import Table
from util.utils import PlannedUnreachableCodeError


class FeatureSelector(Described, ABC):

    @abstractmethod
    def outcome_types(self) -> Sequence[OutcomeType]:
        """Supported outcome types."""
        raise NotImplementedError()

    @abstractmethod
    def ignores_covariates(self) -> bool:
        raise NotImplementedError()


class FeatureSelectorWithPval(FeatureSelector, ABC):

    def __init__(self, descriptor: WithPvalDescriptor):
        FeatureSelector.__init__(self=self, descriptor=descriptor)

    def descriptor(self) -> WithPvalDescriptor:
        res = FeatureSelector.descriptor(self=self)
        assert isinstance(res, WithPvalDescriptor)
        return res

    def p_val(self) -> float:
        return self.descriptor().p_val()

    def _p_val_nick(self) -> str:
        return self.descriptor().p_val_nick()


class SingleFeatureSelector(FeatureSelector, UnivariatePropertyComputer, ABC):
    """A feature selector for a single outcome at a time."""

    @abstractmethod
    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        raise NotImplementedError()

    def selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        if outcome.type() not in self.outcome_types():
            raise ValueError("Input outcome type does not match this feature selector.")
        return self.inner_selection(feature=feature, outcome=outcome)

    def inner_compute_property(self, feature: Sequence[number], outcome: Outcome,
                               covariates: Optional[Table] = None) -> bool:
        return self.inner_selection(feature=feature, outcome=outcome)

    def ignores_covariates(self) -> bool:
        return True

    def descriptor(self) -> SingleFeatureSelectorDescriptor:
        res = UnivariatePropertyComputer.descriptor(self=self)
        assert isinstance(res, SingleFeatureSelectorDescriptor)
        return res

    def __str__(self) -> str:
        return "Unnamed single feature selector"


class SingleFeatureSelectorWithPval(SingleFeatureSelector, FeatureSelectorWithPval, ABC):

    def __init__(self, descriptor: WithPvalDescriptor):
        FeatureSelectorWithPval.__init__(self=self, descriptor=descriptor)


class DummySingleFeatureSelector(SingleFeatureSelector):

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        return True

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical, OutcomeType.survival

    def _create_descriptor(self) -> DummySingleFeatureSelectorDescriptor:
        return DUMMY_SELECTOR_SINGLE_DESCRIPTOR


class MissingSingleFeatureSelector(SingleFeatureSelector):

    def __init__(self, threshold: float = DEFAULT_MISSING_THRESHOLD):
        SingleFeatureSelector.__init__(
            self=self, descriptor=MissingSingleFeatureSelectorDescriptor(threshold=threshold))

    def descriptor(self) -> MissingSingleFeatureSelectorDescriptor:
        res = SingleFeatureSelector.descriptor(self=self)
        assert isinstance(res, MissingSingleFeatureSelectorDescriptor)
        return res

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        n_values = len(feature)
        if n_values == 0:
            return False
        else:
            tot = 0
            for v in feature:
                if math.isnan(v):
                    tot +=1
            return tot/n_values <= self.descriptor().threshold()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return ALL_OUTCOME_TYPES


class MinorFrequencySingleFeatureSelector(SingleFeatureSelector):
    """Checks if the values that are not equal to the most common one are at least as frequent as the threshold."""

    def __init__(self, threshold: float = DEFAULT_MINOR_FREQUENCY_THRESHOLD):
        SingleFeatureSelector.__init__(
            self=self,
            descriptor=MinorFrequencySingleFeatureSelectorDescriptor(threshold=threshold))

    def descriptor(self) -> MinorFrequencySingleFeatureSelectorDescriptor:
        res = SingleFeatureSelector.descriptor(self=self)
        assert isinstance(res, MinorFrequencySingleFeatureSelectorDescriptor)
        return res

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        n_values = len(feature)
        if n_values == 0:
            return False
        else:
            category_counts = Counter(feature)
            if len(category_counts) < 2:
                return False
            else:
                tot = 0
                category_counts = category_counts.most_common()
                for i in range(1, len(category_counts)):
                    label = category_counts[i][0]
                    if not (label is None or math.isnan(label)):
                        tot += category_counts[i][1]
                return tot/n_values >= self.descriptor().threshold()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return ALL_OUTCOME_TYPES


class MAFFeatureSelector(SingleFeatureSelector):
    """Minor allele frequency."""

    def __init__(self, threshold: float = DEFAULT_MAF_THRESHOLD):
        SingleFeatureSelector.__init__(
            self=self,
            descriptor=MAFSingleFeatureSelectorDescriptor(threshold=threshold))

    def descriptor(self) -> MAFSingleFeatureSelectorDescriptor:
        res = SingleFeatureSelector.descriptor(self=self)
        assert isinstance(res, MAFSingleFeatureSelectorDescriptor)
        return res

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        # Allele counts for biallelic SNPs
        count0 = 0
        count1 = 0
        for g in feature:
            if g == 0:
                count0 += 2
            elif g == 1:
                count0 += 1
                count1 += 1
            elif g == 2:
                count1 += 2
            elif g is not None and not pd.isna(g):
                return True  # This is not an SNP, so the filter must not be applied.
        total = count0 + count1
        if total == 0:
            return False
        maf = min(count0, count1) / total
        return maf >= self.descriptor().threshold()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return ALL_OUTCOME_TYPES


class SingleFeatureSelectorAnovaCategorical(SingleFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPval.__init__(
            self=self, descriptor=SingleFeatureSelectorAnovaCategoricalDescriptor(p_val=p_val))

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        return anova_filter_one_feature_checked(x=feature, y=outcome.first_col(), p_val=self.p_val())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,


class SingleFeatureSelectorAnovaSurvival(SingleFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPval.__init__(
            self=self, descriptor=SingleFeatureSelectorAnovaSurvivalDescriptor(p_val=p_val))

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        return anova_filter_one_feature_checked(x=feature, y=survival_events(outcome.data()), p_val=self.p_val())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,


class SingleFeatureSelectorCox(SingleFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPval.__init__(
            self=self, descriptor=SingleFeatureSelectorCoxDescriptor(p_val=p_val))

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        y = outcome.data()
        x = DataFrame(feature)
        return cox_filter_one_feature_raw(x=x, y=y, p_val=self.p_val())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,


class CoxFilterOneColInput:
    def __init__(self, x, y, p_val):
        self.x = x
        self.y = y
        self.p_val = p_val


def cox_filter_one_feature(col_input: CoxFilterOneColInput) -> bool:
    return cox_filter_one_feature_raw(x=col_input.x, y= col_input.y, p_val=col_input.p_val)


def cox_filter_one_feature_raw(x: DataFrame, y: DataFrame, p_val: float) -> bool:
    predictor = LifelinesModel().fit(x=x, y=y)
    if predictor.has_p_vals():
        p_vals = predictor.p_vals()
        return p_vals[0] < p_val
    else:
        return False


class CompositeSingleFeatureSelector(SingleFeatureSelector):
    __categorical_selector: SingleFeatureSelector
    __survival_selector: SingleFeatureSelector

    def __init__(self, categorical_selector: SingleFeatureSelector, survival_selector: SingleFeatureSelector):
        SingleFeatureSelector.__init__(self=self)
        self.__categorical_selector = categorical_selector
        self.__survival_selector = survival_selector

    def _create_descriptor(self) -> CompositeSingleFeatureSelectorDescriptor:
        categorical = self.__categorical_selector.descriptor()
        survival = self.__survival_selector.descriptor()
        return CompositeSingleFeatureSelectorDescriptor(categorical_selector=categorical, survival_selector=survival)

    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:
        if outcome.type() == OutcomeType.categorical:
            return self.__categorical_selector.selection(feature=feature, outcome=outcome)
        elif outcome.type() == OutcomeType.survival:
            return self.__survival_selector.selection(feature=feature, outcome=outcome)
        else:
            raise PlannedUnreachableCodeError()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical, OutcomeType.survival

    def __str__(self) -> str:
        res = "composite single feature selector with\n"
        res += "categorical: " + str(self.__categorical_selector) + "\n"
        res += "survival: " + str(self.__survival_selector) + "\n"
        return res

    def name(self) -> str:
        return "(" + self.__categorical_selector.name() + ", " + self.__survival_selector.name() + ")"

    def nick(self) -> str:
        return "(" + self.__categorical_selector.nick() + "," + self.__survival_selector.nick() + ")"


class JustAnovaSingleFeatureSelector(CompositeSingleFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeSingleFeatureSelector.__init__(
            self=self,
            categorical_selector=SingleFeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=SingleFeatureSelectorAnovaSurvival(p_val=p_val))


class AnovaAndCoxSingleFeatureSelector(CompositeSingleFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeSingleFeatureSelector.__init__(
            self=self,
            categorical_selector=SingleFeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=SingleFeatureSelectorCox(p_val=p_val))


class HWESingleFeatureSelector(SingleFeatureSelectorWithPval):
    """Hardy-Weinberg Equilibrium. If the feature does not appear to be an SNP the test always returns True."""
    __control_class: Optional[Any]

    def __init__(self, control_class: Optional[Any] = None, p_val: float = DEFAULT_HWE_PVAL):
        """If the control_class is None, all samples are used to compute HWE. It makes sense only if the
        samples are extracted at random from the population."""
        SingleFeatureSelectorWithPval.__init__(self=self, descriptor=HWESingleFeatureSelectorDescriptor(p_val=p_val))
        self.__control_class = control_class

    # noinspection PyTypeChecker
    def inner_selection(self, feature: Sequence[number], outcome: Outcome) -> bool:

        hets = 0
        hom1 = 0
        hom2 = 0

        control_class = self.__control_class
        if control_class is None:
            for f in feature:
                if f is None or pd.isna(f):
                    pass
                elif f == 0:
                    hom2 += 1
                elif f == 1:
                    hets += 1
                elif f == 2:
                    hom1 += 1
                else:
                    return True  # Not an SNP, skip HWE test
        else:
            for f, o in zip(feature, outcome.first_col()):
                if f is None or pd.isna(f):
                    pass
                elif f == 0:
                    if o == control_class:
                        hom2 += 1
                elif f == 1:
                    if o == control_class:
                        hets += 1
                elif f == 2:
                    if o == control_class:
                        hom1 += 1
                else:
                    return True  # Not an SNP, skip HWE test
        n_values = hets + hom1 + hom2
        if n_values == 0:
            return False
        else:
            from snphwe import snphwe
            hwe_p = snphwe(hets, hom1, hom2)
            return hwe_p < self.p_val()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,
