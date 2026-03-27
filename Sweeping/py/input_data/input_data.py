from __future__ import annotations

from abc import abstractmethod, ABC
from collections.abc import Sequence, Iterable
from typing import Union, Optional

from pandas import DataFrame, Index

from individual.mv_feature_set_by_names import MVFeatureSetByNames
from input_data.outcome import Outcome

from multi_view_utils import mv_select_by_indices
from util.dataframe.dataframes import prefix_all_cols, columns_in_common
from util.dict_utils import dict_select
from util.list_like import BoolListLike
from util.table.backed_table import BackedTable
from util.table.table_backend.np_table import NpTable
from util.named import NickNamed
import pandas as pd

from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_utils import n_row, n_col
from util.utils import IllegalStateError
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from input_data.model_ready_input_data import ModelReadyInputData
    from util.feature_space_lifter import FeatureSpaceLifterMV



class InputData(NickNamed,ABC):
    """Views contains views, each view being a table, each row being a sample.
        outcomes is a dict of str to Outcome, each element defining one expected output for each of the samples."""
    __views: Views
    __nick: str
    __name: str
    __adjusted_view_def: AdjustedViewDef

    def __init__(self, all_views: Union[dict[str, Union[DataFrame, Table]], Views],
                 nick: str, adjusted_views: Optional[AdjustedViewDef] = None,
                 name: Optional[str] = None):
        """Assuming all views have the same sample at the same row.
        Constructor checks if the views have the same number of samples."""
        new_views = {}
        if isinstance(all_views, Views):
            self.__views = all_views
        else:
            for v in all_views:
                v_data = all_views[v]
                if isinstance(v_data, Table):
                    new_views[v] = v_data
                else:
                    new_views[v] = BackedTable(backend=NpTable(data=v_data.reset_index(drop=True, inplace=False)))
                    # TODO Maybe the project is now ready for not resetting the index.
            self.__views = JustViews(views_dict=new_views)
        self.__nick = nick
        self.__outcomes = {}
        if adjusted_views is None:
            adjusted_views = AdjustedViewDef.create_unadjusted(view_names=self.__views.keys())
        if adjusted_views.all_views_set() != self.__views.keys():
            raise ValueError("Each view must be either adjusted or adjusting.")
        self.__adjusted_view_def = adjusted_views
        if name is None:
            name = nick
        self.__name = name

    def views(self) -> Views:
        return self.__views

    def views_dict(self) -> dict[str, Table]:
        return self.__views.as_dict()

    def views_dict_df(self) -> dict[str, pd.DataFrame]:
        return self.__views.as_dict_df()

    @abstractmethod
    def outcomes(self) -> Sequence[Outcome]:
        """Keeps original order."""
        raise NotImplementedError()

    def outcomes_dict(self) -> dict[str, Outcome]:
        return {o.name(): o for o in self.outcomes()}

    def outcomes_data_dict(self) -> dict[str, DataFrame]:
        res = {}
        for o in self.outcomes():
            res[o.name()] = o.data()
        return res

    def outcome(self, name: str) -> Outcome:
        return self.outcomes_dict()[name]

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    @staticmethod
    def smart_create(
            all_views: Union[dict[str, Union[DataFrame, Table]], Views], outcomes: Iterable[Outcome], nick: str,
            stratify_outcome: Optional[str] = None,
            covariate_views: Optional[Iterable[str]] = None,
            adjusted_views: Optional[AdjustedViewDef] = None,
            name: Optional[str] = None) -> InputData:
        """Selects the subclass to be created in a smart way."""
        outcomes = list(outcomes)
        if covariate_views is not None:
            covariate_views = list(covariate_views)
        if covariate_views is None or not covariate_views:
            if len(outcomes) == 1 and stratify_outcome == outcomes[0].name():
                from input_data.model_ready_input_data import ModelReadyInputData
                return ModelReadyInputData(
                    all_views=all_views, outcome=outcomes[0], nick=nick, adjusted_views=adjusted_views, name=name)
            if len(outcomes) == 0:
                from input_data.evaluation_ready_input_data import NoOutcomesInputData
                return NoOutcomesInputData(all_views=all_views, nick=nick, adjusted_views=adjusted_views, name=name)
        from input_data.full_input_data import FullInputData
        return FullInputData(
            all_views=all_views, outcomes=outcomes, nick=nick, stratify_outcome=stratify_outcome,
            covariate_views=covariate_views, adjusted_views=adjusted_views, name=name)

    @staticmethod
    def create_one_outcome(
            views: Union[dict[str, pd.DataFrame], Views], outcome: Outcome, nick: str,
            covariate_views: Optional[Iterable[str]] = None,
            adjusted_views: Optional[AdjustedViewDef] = None,
            name: Optional[str] = None) -> InputData:
        """Stratify outcome is considered to be the one outcome that is present."""
        return InputData.smart_create(all_views=views, outcomes=[outcome], nick=nick,
                                      stratify_outcome=outcome.name(),
                                      covariate_views=covariate_views, adjusted_views=adjusted_views,
                                      name=name)

    @staticmethod
    def create_no_outcome(
            views: Union[dict[str, pd.DataFrame], Views], nick: str,
            covariate_views: Optional[Iterable[str]] = None,
            adjusted_views: Optional[AdjustedViewDef] = None,
            name: Optional[str] = None) -> InputData:
        from input_data.full_input_data import FullInputData
        return FullInputData(
            all_views=views, outcomes=[], nick=nick, stratify_outcome=None, covariate_views=covariate_views,
            adjusted_views=adjusted_views, name=name)

    def n_views(self) -> int:
        return self.__views.n_views()

    def collapsed_views(self) -> Table:
        """Computed at every call."""
        return self.__views.collapsed()

    def select_outcomes(self, keys: Iterable[str]) -> InputData:
        """Returns a new object, the old one is not modified. Nick remains the same."""
        if self.has_stratify_outcome() and self.stratify_outcome_name() in keys:
            strat = self.stratify_outcome_name()
        else:
            strat = None
        return InputData.smart_create(
            all_views=self.views(), outcomes=list(dict_select(old_dict=self.outcomes_dict(), keys=keys).values()),
            nick=self.nick(), stratify_outcome=strat, covariate_views=self.covariate_view_names(),
            adjusted_views=self.adjusted_view_def(), name=self.name())

    def select_one_outcome(self, outcome_key: str) -> InputData:
        """Returns a new object, the old one is not modified. Nick remains the same."""
        return self.select_outcomes(keys=[outcome_key])

    def collapsed_outcomes(self) -> pd.DataFrame:
        return collapse_outcomes(self.outcomes_dict())

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

    def select_samples(self, row_indices: Sequence[int]) -> InputData:
        """Will return more specific types of InputData in certain cases."""
        res_views = self.views().select_samples(locs=row_indices)
        res_outcomes = [o.select_by_row_indices(indices=row_indices) for o in self.outcomes()]
        return InputData.smart_create(
            all_views=res_views, outcomes=res_outcomes, nick=self.nick(),
            stratify_outcome=self.stratify_outcome_name_optional(), covariate_views=self.covariate_view_names(),
            adjusted_views=self.__adjusted_view_def, name=self.name())

    def serialize(self) -> InputData:
        """Returns a semantically equal object, potentially a smaller version good for serialization."""
        return InputData.smart_create(
            all_views=self.views().serialize(), outcomes=self.outcomes(), nick=self.nick(),
            stratify_outcome=self.stratify_outcome_name_optional(), covariate_views=self.covariate_view_names(),
            adjusted_views=self.__adjusted_view_def, name=self.name())

    def view_names_set(self) -> set[str]:
        """Includes both predictive and adjusting."""
        return self.__adjusted_view_def.all_views_set()

    def view_names_seq(self) -> Sequence[str]:
        """Includes both predictive and adjusting, in alphabetical order."""
        return sorted(self.view_names_set())

    def outcome_names(self) -> Sequence[str]:
        """Keeps original order."""
        return [o.name() for o in self.outcomes()]

    def view(self, view_name: str) -> Table:
        return self.__views[view_name]

    def select_view(self, view_name: str) -> InputData:
        """Returns an InputData object with only the selected view as predictive view and its adjusting views
        that are also included."""
        if view_name in self.covariate_view_names():
            covariate = [view_name]
        else:
            covariate = None
        adjusted_views = AdjustedViewDef(
            view_to_adjusters={view_name: self.adjusted_view_def().adjusters_for_view(view=view_name)})
        return InputData.smart_create(
            all_views={v: self.view(view_name=v) for v in adjusted_views.all_views_seq()}, outcomes=self.outcomes(),
            nick=self.nick(), stratify_outcome=self.stratify_outcome_name_optional(), covariate_views=covariate,
            adjusted_views=adjusted_views, name=self.name())

    @abstractmethod
    def has_stratify_outcome(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def stratify_outcome_name(self) -> str:
        """Default outcome on which to stratify."""
        raise NotImplementedError()

    def stratify_outcome_name_optional(self) -> Optional[str]:
        if self.has_stratify_outcome():
            return self.stratify_outcome_name()
        else:
            return None

    def stratify_outcome(self) -> Outcome:
        return self.outcome(name=self.stratify_outcome_name())

    def stratify_outcome_data(self) -> DataFrame:
        """The data of the outcome that is used for default stratification."""
        return self.stratify_outcome().data()

    def n_outcomes(self) -> int:
        return len(self.outcomes())

    def collapsed_feature_names(self) -> Sequence[str]:
        return self.collapsed_views().colnames()

    def n_samples(self):
        return self.__views.n_samples()

    def standardize_features(self) -> InputData:
        views = self.views_dict_df()
        res_views = {}
        for k in views:
            df = views[k]
            res_views[k] = (df-df.mean()) / df.std()  # Works as long as std is not zero.
        return self.set_views(views=JustViews(views_dict=res_views))

    def _n_samples_consistency(self) -> bool:
        """Checks consistency of sample number between views and outcomes."""
        n = self.n_samples()
        if self.views().n_samples() != n:
            return False
        outcomes = self.outcomes_data_dict()
        for o in outcomes:
            if n_row(outcomes[o]) != n:
                return False
        return True

    def uplift(self, lifter: FeatureSpaceLifterMV) -> InputData:
        res_views = lifter.uplift_views(self.views())
        return self.set_views(views=res_views)

    def collapsed_position(self, view_name: str, feature_name: str) -> int:
        """Returns column number in collapsed views."""
        res = 0
        for vn in self.view_names_seq():
            view = self.view(view_name=vn)
            if vn == view_name:
                index: Index = Index(view.colnames())
                return res + index.get_loc(feature_name)
            else:
                res += n_col(data=view)
        raise ValueError("View not found.")

    def n_features(self) -> int:
        """Considers all views."""
        res = 0
        for vn in self.view_names_seq():
            res += n_col(self.view(view_name=vn))
        return res

    def n_predictive_features(self) -> int:
        """Considers only the features of views directly used for prediction."""
        res = 0
        for vn in self.adjusted_view_def().predictive_view_names_seq():
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
        res += "Name: " + self.name() + "\n"
        res += "Number of samples: " + str(self.n_samples()) + "\n"
        res += "Views (number of columns):\n"
        for vk in self.__views.keys():
            res += str(vk) + " (" + str(n_col(self.__views[vk])) + ")\n"
        res += "Outcomes:\n"
        for o in self.outcomes():
            res += str(o) + "\n"
        if self.has_stratify_outcome():
            res += "Stratify outcome: " + str(self.stratify_outcome_name()) + "\n"
        res += "Adjusted views: " + str(self.__adjusted_view_def) + "\n"
        return res

    def n_features_per_view(self) -> dict[str, int]:
        """Includes both predictive and adjusting, in alphabetical order."""
        return {vn: n_col(self.view(view_name=vn)) for vn in self.view_names_seq()}

    def n_features_per_view_seq(self) -> list[int]:
        """Includes both predictive and adjusting, in alphabetical order."""
        return  [n_col(self.view(view_name=vn)) for vn in self.view_names_seq()]

    def set_views(self, views: Views) -> InputData:
        """The new view names must be a subset of the previous ones. Covariate views and adjusting
        views are preserved as possible. Selects the subclass of InputData to be created in a smart way."""
        new_view_names = set(views.keys())
        if not new_view_names.issubset(self.view_names_set()):
            raise ValueError("New view names must be a subset of the old ones.")
        covariates = set(self.covariate_view_names()).intersection(new_view_names)
        views_to_adjusters = self.__adjusted_view_def.select_views(view_names=new_view_names)
        return InputData.smart_create(
            all_views=views, outcomes=self.outcomes(), nick=self.nick(),
            stratify_outcome=self.stratify_outcome_name_optional(),
            covariate_views=covariates, adjusted_views=views_to_adjusters, name=self.name())

    def select_existing_features(self, features: MVFeatureSetByNames) -> InputData:
        """Only features that are selected will be included. Adjusting features that are not selected will not
        be included."""
        new_views = {}
        for key in features.view_names():
            view = self.view(view_name=key)
            existing_features = set(view.colnames())
            to_fetch = []
            for f in features.view_features(view_name=key):
                if f in existing_features:
                    to_fetch.append(f)
            new_views[key] = view.select_cols_by_names(names=to_fetch)
        return self.set_views(views=JustViews(views_dict=new_views))

    def is_covariate_view(self, view_name: str) -> bool:
        return view_name in self.covariate_view_names()

    def covariates_table(self) -> Table:
        return self.views().select_views(view_names = self.covariate_view_names()).collapsed()

    def has_covariates(self):
        return len(self.covariate_view_names()) > 0

    def set_covariate_views(self, covariate_views: Optional[Iterable[str]] = None) -> InputData:
        stratify_outcome = None
        if self.has_stratify_outcome():
            stratify_outcome = self.stratify_outcome_name()
        return InputData.smart_create(
            all_views=self.views(), outcomes=self.outcomes(), nick=self.nick(), stratify_outcome=stratify_outcome,
            covariate_views=covariate_views, adjusted_views=self.__adjusted_view_def, name=self.name())

    @abstractmethod
    def covariate_view_names(self) -> Sequence[str]:
        raise NotImplementedError()

    def needs_adjustment(self) -> bool:
        """True if there is at least one view that needs adjustment."""
        return self.__adjusted_view_def.needs_adjustment()

    def adjusted_view_def(self) -> AdjustedViewDef:
        return self.__adjusted_view_def

    def predictive_view_names(self) -> Sequence[str]:
        return self.__adjusted_view_def.predictive_view_names_seq()

    def adjuster_view_names(self) -> list[str]:
        """Names are returned in sorted order."""
        return self.__adjusted_view_def.adjuster_view_names()

    def select_active_features(self, active_by_view:  dict[str, Sequence[bool]]) -> InputData:
        """Selects the active features by keeping all the adjusting features."""
        new_views = {}
        for vn in self.predictive_view_names():
            new_views[vn] = self.view(view_name=vn).filter_cols_by_mask(mask=active_by_view[vn])
        for vn in self.adjuster_view_names():
            new_views[vn] = self.view(view_name=vn)
        return self.set_views(views=JustViews(views_dict=new_views))

    def predictive_views(self) -> Views:
        return self.__views.select_views(view_names=self.predictive_view_names())

    def adjuster_views(self) -> Views:
        return self.__views.select_views(view_names=self.adjuster_view_names())

    def select_features(self, masks: dict[str,BoolListLike]) -> InputData:
        """Selects both predictive and adjusting features (adjusting features are included only if specified in the
        masks parameter). Selects the subclass of InputData to be created in a smart way."""
        new_views = {k: self.view(view_name=k).filter_cols_by_mask(mask=v) for k, v in masks.items()}
        return self.set_views(views=JustViews(views_dict=new_views))

    def set_view(self, view_name: str, table: Table) -> InputData:
        """View will be either added or overwritten."""
        return self.set_views(views=self.views().set_view(view_name=view_name, table=table))

    def model_ready(self, outcome: Optional[str] = None) -> ModelReadyInputData:
        """If outcome is not specified and there is only one outcome in this object, then that outcome is used.
        If there are 0 or 2+ outcomes and the outcome is not specified an error is raised."""
        if outcome is None:
            if self.n_outcomes() != 1:
                raise IllegalStateError()
            else:
                outcome = self.outcome_names()[0]
        from input_data.model_ready_input_data import ModelReadyInputData
        return ModelReadyInputData(
            all_views=self.__views, adjusted_views=self.__adjusted_view_def, outcome=self.outcome(name=outcome),
            nick=self.nick(), name=self.name())

    def fast_cols(self) -> InputData:
        return self.set_views(views=self.views().fast_cols())

    def as_cached(self) -> InputData:
        """Will cache the collapsed state of the views."""
        return self.set_views(views=self.views().as_cached())

    def remove_outcomes(self) -> InputData:
        return self.select_outcomes(keys=[])

    def has_non_finite_x(self) -> bool:
        return self.views().has_non_finite()

    def make_all_views_predictive(self) -> InputData:
        return InputData.smart_create(
            all_views=self.views(), outcomes=self.outcomes(), nick=self.nick(),
            stratify_outcome=self.stratify_outcome_name_optional(),
            covariate_views=self.covariate_view_names(),
            adjusted_views=self.__adjusted_view_def.make_all_views_predictive(),
            name=self.name())

    def compile(self, max_cells: int = DEFAULT_MAX_CACHEABLE_CELLS) -> InputData:
        """All views get compiled, including the collapsed view if they are cached views.
        the max_cells is applied to each table separately."""
        return self.set_views(views=self.views().compile(max_cells=max_cells))


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
    a_views = a.views_dict_df()
    b_views = b.views_dict_df()
    res_views_a = {}
    res_views_b = {}
    for k in a_views:
        if k in b_views:
            df_a = a_views[k]
            df_b = b_views[k]
            res_cols = sorted(columns_in_common(df_a, df_b))
            res_views_a[k] = df_a[res_cols]
            res_views_b[k] = df_b[res_cols]
    res_a = a.set_views(views=JustViews(views_dict=res_views_a))
    res_b = b.set_views(views=JustViews(views_dict=res_views_b))
    return res_a, res_b
