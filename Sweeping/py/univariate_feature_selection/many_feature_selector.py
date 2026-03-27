import multiprocessing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from pandas import DataFrame

from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType, ALL_OUTCOME_TYPES
from univariate_feature_selection.parallel_anova import filter_anova_mask
from univariate_feature_selection.single_feature_selector import FeatureSelector, CoxFilterOneColInput, \
    cox_filter_one_feature, SingleFeatureSelectorAnovaCategorical, SingleFeatureSelectorAnovaSurvival, \
    SingleFeatureSelectorCox, SingleFeatureSelector
from univariate_feature_selection.univariate_feature_selector_descriptor import CompositeManyFeatureSelectorDescriptor, \
    DUMMY_SELECTOR_MANY_DESCRIPTOR, DummyManyFeatureSelectorDescriptor, AnovaCategoricalDescriptor, \
    FeatureSelectorCoxDescriptor, ManyFeatureSelectorWithPvalDescriptor, AnovaSurvivalDescriptor, \
    ManyFeatureSelectorClassDescriptor, ManyFeatureSelectorSurvDescriptor, ManyFeatureSelectorFromSingleDescriptor, \
    ManyFeatureSelectorDescriptor, ManyFeatureSelectorPipelineDescriptor
from consts import DEFAULT_P_VAL
from univariate_property_computer.parallel_univariate_property_computer import compute_univariate_property_with_workers
from util.printer.printer import Printer, DEFAULT_PRINTER
from util.sequence_utils import seq_intersection
from util.str_utils import str_paste
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.table.table_utils import n_col
from util.survival.survival_utils import survival_events
from util.utils import PlannedUnreachableCodeError


class ManyFeatureSelector(FeatureSelector):
    """A feature selector for a single view and a single outcome, returns a list of bool for selected features."""

    @abstractmethod
    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        raise NotImplementedError()

    def inner_selection_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        """Override to provide a more optimized version for data frames."""
        return self.inner_selection(
            data=BackedTable(backend=NpTable(data=data)), outcome=outcome, n_proc=n_proc, covariates=covariates,
            printer=printer)

    def selection_mask(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        if outcome.type() not in self.outcome_types():
            raise ValueError("Input outcome type does not match this feature selector.")
        return self.inner_selection(data=data, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)

    def selection_mask_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        if outcome.type() not in self.outcome_types():
            raise ValueError("Input outcome type does not match this feature selector.")
        return self.inner_selection_df(data=data, outcome=outcome, n_proc=n_proc, covariates=covariates,
                                       printer=printer)

    @abstractmethod
    def outcome_types(self) -> Sequence[OutcomeType]:
        """Supported outcome types."""
        raise NotImplementedError()


class DummyManyFeatureSelector(ManyFeatureSelector):

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        return [True]*n_col(data)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return ALL_OUTCOME_TYPES

    def ignores_covariates(self) -> bool:
        return True

    def _create_descriptor(self) -> DummyManyFeatureSelectorDescriptor:
        return DUMMY_SELECTOR_MANY_DESCRIPTOR


class ManyFeatureSelectorWithPval(ManyFeatureSelector, ABC):

    def __init__(self, descriptor: ManyFeatureSelectorWithPvalDescriptor):
        ManyFeatureSelector.__init__(self=self, descriptor=descriptor)

    def descriptor(self) -> ManyFeatureSelectorWithPvalDescriptor:
        res = ManyFeatureSelector.descriptor(self=self)
        assert isinstance(res, ManyFeatureSelectorWithPvalDescriptor)
        return res

    def p_val(self) -> float:
        return self.descriptor().p_val()

    def _p_val_nick(self) -> str:
        return self.descriptor().p_val_nick()


class ManyFeatureSelectorAnovaCategorical(ManyFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPval.__init__(self=self, descriptor=AnovaCategoricalDescriptor(p_val=p_val))

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        inner = ManyFeatureSelectorWithWorkers(single_fs=SingleFeatureSelectorAnovaCategorical(p_val=self.p_val()))
        return inner.inner_selection(data=data, outcome=outcome, n_proc=n_proc)

    def inner_selection_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        return filter_anova_mask(
            view=data, outcome=outcome.first_col(), n_proc=n_proc, p_val=self.p_val(), printer=printer)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,

    def ignores_covariates(self) -> bool:
        return True


class ManyFeatureSelectorAnovaSurvival(ManyFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPval.__init__(self=self, descriptor=AnovaSurvivalDescriptor(p_val=p_val))

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        inner = ManyFeatureSelectorWithWorkers(single_fs=SingleFeatureSelectorAnovaSurvival(p_val=self.p_val()))
        return inner.inner_selection(data=data, outcome=outcome, n_proc=n_proc)

    def inner_selection_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        return filter_anova_mask(
            view=data, outcome=survival_events(outcome.data()), n_proc=n_proc, p_val=self.p_val(), printer=printer)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,

    def ignores_covariates(self) -> bool:
        return True


class ManyFeatureSelectorCox(ManyFeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPval.__init__(self=self, descriptor=FeatureSelectorCoxDescriptor(p_val=p_val))

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        inner = ManyFeatureSelectorWithWorkers(single_fs=SingleFeatureSelectorCox(p_val=self.p_val()))
        return inner.inner_selection(data=data, outcome=outcome, n_proc=n_proc)

    def inner_selection_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        n_cols = n_col(data)
        y = outcome.data()
        inputs = (CoxFilterOneColInput(x=data.loc[:, [c]], y=y, p_val=self.p_val())
                  for c in data.columns)
        cpu_count = multiprocessing.cpu_count()
        proc_to_use = max(1, min(n_proc, cpu_count, n_cols))
        if proc_to_use == 1:
            return [cox_filter_one_feature(col_input=i) for i in inputs]
        else:
            with ProcessPoolExecutor(max_workers=proc_to_use) as workers_pool:
                res = workers_pool.map(cox_filter_one_feature, inputs, chunksize=16)
                return list(res)

    def ignores_covariates(self) -> bool:
        return True

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,


class CompositeManyFeatureSelector(ManyFeatureSelector):
    __categorical_selector: ManyFeatureSelector
    __survival_selector: ManyFeatureSelector

    def __init__(self, categorical_selector: ManyFeatureSelector, survival_selector: ManyFeatureSelector):
        ManyFeatureSelector.__init__(self=self)
        self.__categorical_selector = categorical_selector
        self.__survival_selector = survival_selector

    def _create_descriptor(self) -> CompositeManyFeatureSelectorDescriptor:
        categorical_selector = self.__categorical_selector.descriptor()
        survival_selector = self.__survival_selector.descriptor()
        assert isinstance(categorical_selector, ManyFeatureSelectorClassDescriptor)
        assert isinstance(survival_selector, ManyFeatureSelectorSurvDescriptor)
        return CompositeManyFeatureSelectorDescriptor(
            categorical_selector=categorical_selector,
            survival_selector=survival_selector)

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        if outcome.type() == OutcomeType.categorical:
            return self.__categorical_selector.selection_mask(
                data=data, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)
        elif outcome.type() == OutcomeType.survival:
            return self.__survival_selector.selection_mask(
                data=data, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)
        else:
            raise PlannedUnreachableCodeError()

    def inner_selection_df(
            self, data: DataFrame, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        if outcome.type() == OutcomeType.categorical:
            return self.__categorical_selector.selection_mask_df(
                data=data, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)
        elif outcome.type() == OutcomeType.survival:
            return self.__survival_selector.selection_mask_df(
                data=data, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)
        else:
            raise PlannedUnreachableCodeError()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical, OutcomeType.survival

    def ignores_covariates(self) -> bool:
        return self.__categorical_selector.ignores_covariates() and self.__survival_selector.ignores_covariates()

    def __str__(self) -> str:
        res = "composite many feature selector with\n"
        res += "categorical: " + str(self.__categorical_selector) + "\n"
        res += "survival: " + str(self.__survival_selector) + "\n"
        return res

    def name(self) -> str:
        return "(" + self.__categorical_selector.name() + ", " + self.__survival_selector.name() + ")"

    def nick(self) -> str:
        return "(" + self.__categorical_selector.nick() + "," + self.__survival_selector.nick() + ")"


class JustAnovaManyFeatureSelector(CompositeManyFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeManyFeatureSelector.__init__(
            self=self,
            categorical_selector=ManyFeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=ManyFeatureSelectorAnovaSurvival(p_val=p_val))


class AnovaAndCoxManyFeatureSelector(CompositeManyFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeManyFeatureSelector.__init__(
            self=self,
            categorical_selector=ManyFeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=ManyFeatureSelectorCox(p_val=p_val))


class ManyFeatureSelectorWithWorkers(ManyFeatureSelector):
    __single_fs: SingleFeatureSelector

    def __init__(self, single_fs: SingleFeatureSelector):
        ManyFeatureSelector.__init__(self=self)
        self.__single_fs = single_fs

    def inner_selection(
            self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
            printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        return compute_univariate_property_with_workers(
            single_feature_computer=self.__single_fs, data=data, outcome=outcome,
            n_proc=n_proc, task_name="Feature selection", covariates=covariates, printer=printer)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return self.__single_fs.outcome_types()

    def ignores_covariates(self) -> bool:
        return self.__single_fs.ignores_covariates()

    def __str__(self) -> str:
        return "Many feature selector with workers using " + str(self.__single_fs)

    def nick(self) -> str:
        return self.__single_fs.nick()

    def _create_descriptor(self) -> ManyFeatureSelectorFromSingleDescriptor:
        return ManyFeatureSelectorFromSingleDescriptor(single_fs=self.__single_fs.descriptor())


class ManyFeatureSelectorPipeline(ManyFeatureSelector):
    __selectors: Sequence[ManyFeatureSelector]

    def __init__(self, selectors: Sequence[ManyFeatureSelector]):
        ManyFeatureSelector.__init__(self=self)
        self.__selectors = list(selectors)

    def _create_descriptor(self) -> ManyFeatureSelectorPipelineDescriptor:
        selectors = []
        for s in self.__selectors:
            d = s.descriptor()
            assert isinstance(d, ManyFeatureSelectorDescriptor)
            selectors.append(d)
        return ManyFeatureSelectorPipelineDescriptor(selectors=selectors)

    def inner_selection(self, data: Table, outcome: Outcome, n_proc: int = 1, covariates: Optional[Table] = None,
                        printer: Printer = DEFAULT_PRINTER) -> list[bool]:
        printer.print("Starting feature selection pipeline.")
        n_cols = data.n_col()
        printer.print_variable("initial features", n_cols)
        kept = list(range(n_cols))
        for s in self.__selectors:
            printer.print("Applying feature selector " + s.name())
            temp_table = data.select_cols(kept)
            s_mask = s.selection_mask(
                data=temp_table, outcome=outcome, n_proc=n_proc, covariates=covariates, printer=printer)
            kept = [kept[i] for i in range(len(s_mask)) if s_mask[i]]
            printer.print_variable("remaining features", len(kept))
        res = [False for _ in range(n_cols)]
        for p in kept:
            res[p] = True
        printer.print("Feature selection pipeline finished.")
        return res

    def outcome_types(self) -> Sequence[OutcomeType]:
        n_selectors = len(self.__selectors)
        if n_selectors == 0:
            return ALL_OUTCOME_TYPES
        else:
            return seq_intersection([s.outcome_types() for s in self.__selectors])

    def ignores_covariates(self) -> bool:
        for s in self.__selectors:
            if not s.ignores_covariates():
                return False
        return True

    def nick(self) -> str:
        return str_paste(parts=[s.nick() for s in self.__selectors], separator="_")


DUMMY_SELECTOR_MANY = DummyManyFeatureSelector()