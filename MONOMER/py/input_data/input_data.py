from __future__ import annotations

from collections.abc import Sequence

from pandas import DataFrame
from input_data.outcome import Outcome
from multi_view_utils import collapse_views, mv_select_by_indices
from util.dataframes import prefix_all_cols, columns_in_common, n_col, n_row
from util.named import NickNamed
import pandas as pd
from util.utils import dict_select, IllegalStateError
from views.views import Views, JustViews


class InputData(NickNamed):
    """Views is a dict of views, each view being a dataframe, each row being a sample.
        y is a sequence of anything, each element being an expected output."""
    __views: dict[str, pd.DataFrame]
    __outcomes: dict[str, Outcome]
    __stratify_outcome: str
    __nick: str

    def __init__(self, views: dict[str, pd.DataFrame], outcomes: Sequence[Outcome],
                 nick: str, stratify_outcome: str = None):
        """Assuming all views have the same sample at the same row."""
        new_views = {}
        for v in views:
            new_views[v] = DataFrame(views[v]).reset_index(drop=True)
        self.__views = new_views
        self.__nick = nick
        self.__outcomes = {}
        for o in outcomes:
            self.__outcomes[o.name()] = o
        self.__stratify_outcome = stratify_outcome
        if not self.__n_samples_consistency():
            raise ValueError("Number of samples is not consistent.\n" + str(self))

    def views(self) -> dict[str, pd.DataFrame]:
        return self.__views

    def outcomes(self) -> Sequence[Outcome]:
        return list(self.__outcomes.values())

    def outcomes_dict(self) -> dict[str, Outcome]:
        return self.__outcomes

    def outcomes_data_dict(self) -> dict[str, DataFrame]:
        res = {}
        for o in self.outcomes():
            res[o.name()] = o.data()
        return res

    def outcome(self, name: str) -> Outcome:
        return self.__outcomes[name]

    def __outcomes_data(self) -> [DataFrame]:
        return [self.__outcomes[k].data() for k in self.__outcomes.keys()]

    def nick(self) -> str:
        return self.__nick

    @staticmethod
    def create_one_outcome(
            views: dict[str, pd.DataFrame], outcome: Outcome, nick: str) -> InputData:
        name = outcome.name()
        return InputData(views=views, outcomes=[outcome], nick=nick, stratify_outcome=name)

    def n_views(self) -> int:
        return len(self.views())

    def collapsed_views(self) -> pd.DataFrame:
        """Computed at every call."""
        return collapse_views(self.views())

    def select_outcomes(self, keys: [str]) -> InputData:
        """Returns a new object, the old one is not modified. Nick remains the same."""
        if self.__stratify_outcome in keys:
            strat = self.__stratify_outcome
        else:
            strat = None
        return InputData(
            views=self.views(), outcomes=list(dict_select(old_dict=self.outcomes_dict(), keys=keys).values()),
            nick=self.nick(), stratify_outcome=strat)

    def collapsed_outcomes(self) -> pd.DataFrame:
        return collapse_outcomes(self.__outcomes)

    def x(self) -> Views:
        return JustViews(views_dict=self.views())

    def select_all_sets(self, train_indices: Sequence[int], test_indices: Sequence[int]) -> tuple[Views, dict[str, DataFrame], Views, dict[str, DataFrame]]:
        """ Selects all sets of samples for the passed fold. """
        views = self.views()
        y = self.outcomes_data_dict()
        x_train = JustViews(views_dict=mv_select_by_indices(views, train_indices))
        y_train = mv_select_by_indices(y, train_indices)
        x_test = JustViews(views_dict=mv_select_by_indices(views, test_indices))
        y_test = mv_select_by_indices(y, test_indices)
        return x_train, y_train, x_test, y_test

    def select_samples(self, row_indices: [int]) -> InputData:
        res_views = mv_select_by_indices(self.views(), row_indices)
        res_outcomes = [o.select_by_row_indices(indices=row_indices) for o in self.outcomes()]
        return InputData(views=res_views, outcomes=res_outcomes, nick=self.nick(),
                         stratify_outcome=self.__stratify_outcome)

    def view_names(self) -> [str]:
        return list(self.__views.keys())

    def outcome_names(self) -> [str]:
        return list(self.__outcomes.keys())

    def view(self, view_name: str) -> pd.DataFrame:
        return self.__views[view_name]

    def select_view(self, view_name: str) -> InputData:
        return InputData(
            views={view_name: self.view(view_name=view_name)}, outcomes=self.outcomes(),
            nick=self.nick(), stratify_outcome=self.__stratify_outcome)

    def stratify_outcome_name(self) -> str:
        """Default outcome on which to stratify."""
        if self.__stratify_outcome is not None:
            return self.__stratify_outcome
        else:
            raise IllegalStateError()

    def stratify_outcome(self) -> Outcome:
        return self.__outcomes[self.stratify_outcome_name()]

    def stratify_outcome_data(self) -> DataFrame:
        return self.stratify_outcome().data()

    def n_outcomes(self) -> int:
        return len(self.__outcomes)

    def collapsed_feature_names(self) -> Sequence[str]:
        return self.collapsed_views().columns

    def n_samples(self):
        keys = self.__views.keys()
        if len(keys) == 0:
            return 0
        else:
            return self.__views[next(iter(keys))].shape[0]

    def standardize_features(self) -> InputData:
        views = self.views()
        res_views = {}
        for k in views:
            df = views[k]
            res_views[k] = (df-df.mean()) / df.std()  # Works as long as std is not zero.
        return InputData(
            views=res_views, outcomes=self.outcomes(), nick=self.nick(), stratify_outcome=self.stratify_outcome_name())

    def __n_samples_consistency(self) -> bool:
        n = self.n_samples()
        views = self.views()
        for v in views:
            if n_row(views[v]) != n:
                return False
        outcomes = self.outcomes_data_dict()
        for o in outcomes:
            if n_row(outcomes[o]) != n:
                return False
        return True

    def __str__(self) -> str:
        res = "Nick: " + self.nick() + "\n"
        res += "Number of samples: " + str(self.n_samples()) + "\n"
        res += "Views (number of columns):\n"
        for vk in self.__views:
            res += vk + " (" + str(n_col(self.__views[vk])) + ")\n"
        res += "Outcomes:\n"
        for o in self.__outcomes:
            res += str(self.__outcomes[o]) + "\n"
        res += "Stratify outcome: " + str(self.__stratify_outcome) + "\n"
        return res


def collapse_outcomes(outcomes: dict[str, Outcome]) -> DataFrame:
    outcome_keys = list(outcomes.keys())
    n_outcomes = len(outcome_keys)
    if n_outcomes == 0:
        return pd.DataFrame()
    res = prefix_all_cols(outcomes[outcome_keys[0]].data(), "0_")  # Assuming there is at least a view
    for i in range(1, n_outcomes):
        k = outcome_keys[i]
        res = pd.concat([res, prefix_all_cols(outcomes[k].data(), str(i)+"_")], axis=1)
    return res


def select_common_features(a: InputData, b: InputData) -> tuple[InputData, InputData]:
    a_views = a.views()
    b_views = b.views()
    res_views_a = {}
    res_views_b = {}
    for k in a_views:
        if k in b_views:
            df_a = a_views[k]
            df_b = b_views[k]
            res_cols = sorted(columns_in_common(df_a, df_b))
            res_views_a[k] = df_a[res_cols]
            res_views_b[k] = df_b[res_cols]
    res_a = InputData(
        views=res_views_a, outcomes=a.outcomes(), nick=a.nick(), stratify_outcome=a.stratify_outcome_name())
    res_b = InputData(
        views=res_views_b, outcomes=b.outcomes(), nick=b.nick(), stratify_outcome=b.stratify_outcome_name())
    return res_a, res_b
