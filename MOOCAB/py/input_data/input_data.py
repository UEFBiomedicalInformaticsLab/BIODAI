from __future__ import annotations

from collections.abc import Sequence, Iterable
from typing import Union, Optional

from pandas import DataFrame

from individual.mv_feature_set_by_names import MVFeatureSetByNames
from input_data.outcome import Outcome

from multi_view_utils import mv_select_by_indices
from util.dataframes import prefix_all_cols, columns_in_common, n_col, n_row
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.named import NickNamed
import pandas as pd
from util.utils import dict_select, IllegalStateError
from views.views import Views, JustViews


class InputData(NickNamed):
    """Views contains views, each view being a dataframe, each row being a sample.
        outcomes is a dict of str to Outcome, each element defining one expected output for each of the samples."""
    __views: Views
    __outcomes: dict[str, Outcome]
    __stratify_outcome: Optional[str]
    __nick: str

    def __init__(self, views: Union[dict[str, pd.DataFrame], Views], outcomes: Sequence[Outcome],
                 nick: str, stratify_outcome: str = None):
        """Assuming all views have the same sample at the same row.
        Constructor checks if the views have the same number of samples."""
        new_views = {}
        if isinstance(views, Views):
            self.__views = views
        else:
            for v in views:
                new_views[v] = DataFrame(views[v]).reset_index(drop=True)
            self.__views = JustViews(views_dict=new_views)
        self.__nick = nick
        self.__outcomes = {}
        for o in outcomes:
            self.__outcomes[o.name()] = o
        self.__stratify_outcome = stratify_outcome
        if not self.__n_samples_consistency():
            raise ValueError("Number of samples is not consistent.\n" + str(self))

    def views(self) -> Views:
        return self.__views

    def views_dict(self) -> dict[str, pd.DataFrame]:
        return self.__views.as_dict()

    def outcomes(self) -> Sequence[Outcome]:
        """Keeps original order."""
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

    @staticmethod
    def create_no_outcome(
            views: dict[str, pd.DataFrame], nick: str) -> InputData:
        return InputData(views=views, outcomes=[], nick=nick, stratify_outcome=None)

    def n_views(self) -> int:
        return self.__views.n_views()

    def collapsed_views(self) -> pd.DataFrame:
        """Computed at every call."""
        return self.__views.collapsed()

    def select_outcomes(self, keys: Iterable[str]) -> InputData:
        """Returns a new object, the old one is not modified. Nick remains the same."""
        if self.__stratify_outcome in keys:
            strat = self.__stratify_outcome
        else:
            strat = None
        return InputData(
            views=self.views(), outcomes=list(dict_select(old_dict=self.outcomes_dict(), keys=keys).values()),
            nick=self.nick(), stratify_outcome=strat)

    def select_one_outcome(self, outcome_key: str) -> InputData:
        """Returns a new object, the old one is not modified. Nick remains the same."""
        return self.select_outcomes(keys=[outcome_key])

    def collapsed_outcomes(self) -> pd.DataFrame:
        return collapse_outcomes(self.__outcomes)

    def x(self) -> Views:
        return JustViews(views_dict=self.views_dict())

    def select_all_sets(
            self, train_indices: Sequence[int],
            test_indices: Sequence[int]) -> tuple[Views, dict[str, DataFrame], Views, dict[str, DataFrame]]:
        """ Selects all sets of samples for the passed fold. """
        y = self.outcomes_data_dict()
        views = self.views()
        x_train = views.select_samples(locs=train_indices)
        y_train = mv_select_by_indices(y, train_indices)
        x_test = views.select_samples(locs=test_indices)
        y_test = mv_select_by_indices(y, test_indices)
        return x_train, y_train, x_test, y_test

    def select_samples(self, row_indices: [int]) -> InputData:
        res_views = self.views().select_samples(locs=row_indices)
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
        """The data of the outcome that is used for default stratification."""
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
        views = self.views_dict()
        res_views = {}
        for k in views:
            df = views[k]
            res_views[k] = (df-df.mean()) / df.std()  # Works as long as std is not zero.
        return InputData(
            views=res_views, outcomes=self.outcomes(), nick=self.nick(), stratify_outcome=self.__stratify_outcome)

    def __n_samples_consistency(self) -> bool:
        n = self.n_samples()
        if self.views().n_samples() != n:
            return False
        outcomes = self.outcomes_data_dict()
        for o in outcomes:
            if n_row(outcomes[o]) != n:
                return False
        return True

    def uplift(self, lifter: FeatureSpaceLifterMV) -> InputData:
        res_views = lifter.uplift_dict(self.views_dict())
        return InputData(
            views=res_views,
            outcomes=self.outcomes(),
            nick=self.nick(),
            stratify_outcome=self.__stratify_outcome)

    def collapsed_position(self, view_name: str, feature_name: str) -> int:
        """Returns column number in collapsed views."""
        res = 0
        for vn in self.view_names():
            view = self.view(view_name=vn)
            if vn == view_name:
                return res + view.columns.get_loc(feature_name)
            else:
                res += n_col(df=view)
        raise ValueError("View not found.")

    def n_features(self) -> int:
        res = 0
        for vn in self.view_names():
            res += n_col(self.view(view_name=vn))
        return res

    def get_mask(self, features_by_names: MVFeatureSetByNames) -> list[bool]:
        res = [False]*self.n_features()
        for view in features_by_names.view_names():
            for f in features_by_names.view_features(view_name=view):
                res[self.collapsed_position(view_name=view, feature_name=f)] = True
        return res

    def __str__(self) -> str:
        res = "Nick: " + self.nick() + "\n"
        res += "Number of samples: " + str(self.n_samples()) + "\n"
        res += "Views (number of columns):\n"
        for vk in self.__views.keys():
            res += str(vk) + " (" + str(n_col(self.__views[vk])) + ")\n"
        res += "Outcomes:\n"
        for o in self.__outcomes:
            res += str(self.__outcomes[o]) + "\n"
        res += "Stratify outcome: " + str(self.__stratify_outcome) + "\n"
        return res

    def n_features_per_view(self) -> list[int]:
        res = []
        for vn in self.view_names():
            res.append(n_col(self.view(view_name=vn)))
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
    a_views = a.views_dict()
    b_views = b.views_dict()
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
