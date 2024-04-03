import multiprocessing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from pandas import DataFrame

from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from model.survival_model import LifelinesModel
from univariate_feature_selection.parallel_anova import filter_anova_mask, DEFAULT_P_VAL
from util.dataframes import n_col
from util.survival.survival_utils import survival_events
from util.utils import PlannedUnreachableCodeError


class FeatureSelector (ABC):
    """A feature selector for a single view and a single outcome, returns a list of bool for selected features."""

    @abstractmethod
    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        raise NotImplementedError()

    def selection_mask(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        if outcome.type() not in self.outcome_types():
            raise ValueError("Input outcome type does not match this feature selector.")
        return self.inner_selection(view=view, outcome=outcome, n_proc=n_proc)

    @abstractmethod
    def outcome_types(self) -> Sequence[OutcomeType]:
        """Supported outcome types."""
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Unnamed feature selector"


class DummyFeatureSelector(FeatureSelector):

    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        return [True]*n_col(view)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical, OutcomeType.survival

    def __str__(self) -> str:
        return "Dummy feature selector"


class FeatureSelectorWithPval(FeatureSelector, ABC):
    __p_val: float

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        self.__p_val = p_val

    def p_val(self) -> float:
        return self.__p_val

    def __str__(self) -> str:
        return "Unnamed feature selector with p-value"


class FeatureSelectorAnovaCategorical(FeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        FeatureSelectorWithPval.__init__(self=self, p_val=p_val)

    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        return filter_anova_mask(view=view, outcome=outcome.fist_col(), n_proc=n_proc, p_val=self.p_val())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,

    def __str__(self) -> str:
        return "feature selector any na and anova"


class FeatureSelectorAnovaSurvival(FeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        FeatureSelectorWithPval.__init__(self=self, p_val=p_val)

    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        return filter_anova_mask(view=view, outcome=survival_events(outcome.data()), n_proc=n_proc, p_val=self.p_val())

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,

    def __str__(self) -> str:
        return "feature selector any na and anova on survival events"


class CoxFilterOneColInput:
    def __init__(self, x, y, p_val):
        self.x = x
        self.y = y
        self.p_val = p_val


def cox_filter_one_feature(col_input: CoxFilterOneColInput) -> bool:
    predictor = LifelinesModel().fit(x=col_input.x, y=col_input.y)
    if predictor.has_p_vals():
        p_vals = predictor.p_vals()
        return p_vals[0] < col_input.p_val
    else:
        return False


class FeatureSelectorCox(FeatureSelectorWithPval):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        FeatureSelectorWithPval.__init__(self=self, p_val=p_val)

    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        n_cols = n_col(view)
        y = outcome.data()
        inputs = [CoxFilterOneColInput(x=view.loc[:, [c]], y=y, p_val=self.p_val())
                  for c in view.columns]
        cpu_count = multiprocessing.cpu_count()
        proc_to_use = max(1, min(n_proc, cpu_count, n_cols))
        if proc_to_use == 1:
            return [cox_filter_one_feature(col_input=i) for i in inputs]
        else:
            with ProcessPoolExecutor(max_workers=proc_to_use) as workers_pool:
                res = workers_pool.map(cox_filter_one_feature, inputs, chunksize=16)
                return list(res)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,

    def __str__(self) -> str:
        return "feature selector Cox"


class CompositeFeatureSelector(FeatureSelector):
    __categorical_selector: FeatureSelector
    __survival_selector: FeatureSelector

    def __init__(self, categorical_selector: FeatureSelector, survival_selector: FeatureSelector):
        self.__categorical_selector = categorical_selector
        self.__survival_selector = survival_selector

    def inner_selection(self, view: DataFrame, outcome: Outcome, n_proc: int = 1) -> list[bool]:
        if outcome.type() == OutcomeType.categorical:
            return self.__categorical_selector.selection_mask(view=view, outcome=outcome, n_proc=n_proc)
        elif outcome.type() == OutcomeType.survival:
            return self.__survival_selector.selection_mask(view=view, outcome=outcome, n_proc=n_proc)
        else:
            raise PlannedUnreachableCodeError()

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical, OutcomeType.survival

    def __str__(self) -> str:
        res = "composite feature selector with\n"
        res += "categorical: " + str(self.__categorical_selector) + "\n"
        res += "survival: " + str(self.__survival_selector) + "\n"
        return res


class JustAnovaFeatureSelector(CompositeFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeFeatureSelector.__init__(
            self=self,
            categorical_selector=FeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=FeatureSelectorAnovaSurvival(p_val=p_val))


class AnovaAndCoxFeatureSelector(CompositeFeatureSelector):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        CompositeFeatureSelector.__init__(
            self=self,
            categorical_selector=FeatureSelectorAnovaCategorical(p_val=p_val),
            survival_selector=FeatureSelectorCox(p_val=p_val))
