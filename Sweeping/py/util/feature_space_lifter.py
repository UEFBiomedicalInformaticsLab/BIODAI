from __future__ import annotations
from collections.abc import Sequence

from pandas import DataFrame

from hyperparam_manager.mv_hyperparam_manager.abstract_mv_hp_manager import add_adjusting_views
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from util.dataframe.dataframes import select_cols_by_mask
from util.dict_utils import keys_are_sorted
from util.list_like import BoolListLike
from util.sequence_utils import flatten_iterable_of_iterable
from util.select_from_sequence import filter_by_booleans
from util.sparse_bool_list_by_set import SparseBoolList, SparseBoolListBySet
from util.table.table import Table
from util.table.table_utils import is_2d
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews


class FeatureSpaceLifter:
    """Applies to a single view."""
    __active_features: SparseBoolList  # Features in bigger space that are in smaller space.

    def __init__(self, active_features: Sequence[bool]):
        if isinstance(active_features, SparseBoolList):
            self.__active_features = active_features
        else:
            self.__active_features = SparseBoolListBySet(seq=active_features)

    def uplift(self, features: Sequence[bool]) -> list[bool]:
        """Features from bigger space to smaller space.
        It checks that the features sequence has the correct size for this lifter."""
        return filter_by_booleans(data=features, selectors=self.__active_features)

    def downlift(self, features: Sequence[bool]) -> BoolListLike:
        """Features from smaller space to bigger space."""
        if not isinstance(features, Sequence):
            raise ValueError("features not sequence. Features: " + str(features))
        af = self.__active_features
        big_len = len(af)
        res = SparseBoolListBySet(min_size=big_len)
        j = 0
        for i in af.true_positions():
            try:
                if features[j]:
                    res.set_true(i)
            except IndexError as e:
                raise IndexError(
                    "i: " + str(i) + "\n" +
                    "j: " + str(j) + "\n" +
                    "self features len: " + str(big_len) + "\n" +
                    "features type: " + str(type(features)) + "\n" +
                    "features: " + str(features) + "\n" +
                    "original error: " + str(e) + "\n")
            j += 1
        return res

    def uplift_df(self, df: DataFrame) -> DataFrame:
        """Columns from bigger space to smaller space.
        Raises an exception if the active features mask is of wrong size."""
        if not is_2d(df):
            raise ValueError("DataFrame is not 2D.\n" + "Passed dataframe shape: " + str(df.shape) + "\n")
        return select_cols_by_mask(df=df, mask=self.__active_features)

    def uplift_table(self, table: Table) -> Table:
        """Columns from bigger space to smaller space.
        Checks for size congruence."""
        return table.filter_cols_by_mask(mask=self.__active_features)

    def active_features_mask(self) -> Sequence[bool]:
        """Mask of features in bigger space that are in smaller space."""
        return self.__active_features

    def lower_space_size(self) -> int:
        return len(self.__active_features)

    def __str__(self) -> str:
        return str(self.__active_features)


class FeatureSpaceLifterMV:
    """Views are sorted alphabetically."""
    __single_view_lifters: dict[str,FeatureSpaceLifter]

    def __init__(self, single_view_lifters: dict[str,FeatureSpaceLifter]):
        """There must be lifters also for views that are not used in the upper space, otherwise it is not
        possible to properly downlift."""
        if not keys_are_sorted(single_view_lifters):
            raise ValueError("views must be in alphabetical order.")
        self.__single_view_lifters = dict(single_view_lifters)

    def uplift(self, features_mv: dict[str,Sequence[bool]]) -> dict[str,list[bool]]:
        """Features from bigger space to smaller space."""
        return {v: self.__single_view_lifters[v].uplift(features=features) for v, features in features_mv.items()}

    def downlift(self, features_mv: dict[str,Sequence[bool]]) -> dict[str,Sequence[bool]]:
        """Features from smaller space to bigger space."""
        res = {}
        for v, lifter in self.__single_view_lifters.items():
            if v in features_mv:
                res[v] = lifter.downlift(features=features_mv[v])
            else:
                res[v] = SparseBoolListBySet(min_size=lifter.lower_space_size())
        return res

    def uplift_dfs(self, dfs: dict[str,DataFrame]) -> dict[str,DataFrame]:
        """Columns from bigger space to smaller space. Uplifts a dictionary [name, dataframe]."""
        res = {}
        for v, df in dfs.items():
            if not is_2d(df):
                raise ValueError("DataFrame is not 2D.\n" + "Passed dataframe shape: " + str(df.shape) + "\n")
            res[v] = self.__single_view_lifters[v].uplift_df(df=df)
        return res

    def collapse(self) -> FeatureSpaceLifter:
        return FeatureSpaceLifter(
            active_features=flatten_iterable_of_iterable(
                x=[lifter.active_features_mask() for lifter in self.__single_view_lifters.values()]))

    def uplift_views(self, views: Views) -> Views:
        """Views from bigger space to smaller space."""
        res = {}
        for v in views.keys():
            res[v] = self.__single_view_lifters[v].uplift_table(views[v])
        return JustViews(views_dict=res)

    def lower_space_dummy_mv_hp_manager(self) ->MvHyperparamManager:
        """Returned hp manager has just predictive views, no adjusting ones. All the features in the lower space
        are active."""
        return MaskMvHpManager(
            adj_view_def=AdjustedViewDef.create_unadjusted(self.__single_view_lifters.keys()),
            view_col_numbers={k: len(v.active_features_mask()) for k,v in self.__single_view_lifters.items()})

    @classmethod
    def create_from_active_features_and_adjusted_view_def(
            cls, active_by_view:  dict[str, Sequence[bool]],
            adjusted_view_def: AdjustedViewDef) -> FeatureSpaceLifterMV:
        active_by_view = add_adjusting_views(current_selection=active_by_view, adj_view_def=adjusted_view_def)
        single_view_lifters = {}
        for view_name in active_by_view:
            single_view_lifters[view_name] = FeatureSpaceLifter(active_features=active_by_view[view_name])
        return FeatureSpaceLifterMV(single_view_lifters=single_view_lifters)

