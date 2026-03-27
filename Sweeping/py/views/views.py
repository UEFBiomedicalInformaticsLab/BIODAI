from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Set, Sequence, Iterable
from typing import Optional, Union

import numpy as np
from frozenlist import FrozenList
from pandas import DataFrame

from util.table.backed_table import BackedTable
from util.table.table_backend.np_table import NpTable
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_utils import n_col, n_row
from util.list_like import ListLike
from util.table.collapsed_tables import FreshCollapsedTables
from util.table.table import Table


class Views(ABC):
    """Views are ordered alphabetically."""

    @abstractmethod
    def keys(self) -> Set[str]:
        """Views are ordered alphabetically."""
        raise NotImplementedError()

    @abstractmethod
    def view(self, key: Union[str, int]) -> Table:
        """If receiving a string it is used as a key, if receiving an int it is used as a position.
        Allows use in zip."""
        raise NotImplementedError()

    @abstractmethod
    def collapsed(self) -> Table:
        raise NotImplementedError()

    def to_dataframe(self) -> DataFrame:
        """Shorthand for collapsed().to_dataframe()"""
        return self.collapsed().to_dataframe()

    def dataframes(self) -> Sequence[DataFrame]:
        return [self.view(key=i).to_dataframe() for i in range(self.n_views())]

    def __getitem__(self, key: Union[str, int]) -> Table:
        """If receiving a string it is used as a key, if receiving an int it is used as a position.
        Allows use in zip."""
        return self.view(key=key)

    def __str__(self) -> str:
        res = ""
        res += "Views:\n"
        for vk in self.keys():
            ncol = n_col(self[vk])
            res += vk
            if ncol > 10:
                res += " (" + str(n_col(self[vk])) + " columns)\n"
            else:
                res += " " + str(self[vk].colnames()) + "\n"
        return res

    @abstractmethod
    def as_cached(self) -> CachedViews:
        """Cached views cache the collapsed state."""
        raise NotImplementedError()

    def collapsed_filtered_by_mask(self, mask: ListLike) -> Table:
        """The views are collapsed (but not compiled), then filtered. The mask is applied to the columns."""
        return self.collapsed().filter_cols_by_mask(mask=mask)

    def as_dict(self) -> dict[str, Table]:
        res = {}
        for k in self.keys():
            res[k] = self.view(key=k)
        return res

    def as_dict_df(self) -> dict[str, DataFrame]:
        res = {}
        for k in self.keys():
            res[k] = self.view(key=k).to_dataframe()
        return res

    def n_samples(self) -> int:
        for k in self.keys():
            return n_row(self.view(key=k))
        return 0  # If there are no views.

    def n_features(self) -> int:
        return sum((n_col(self.view(key=k)) for k in self.keys()))

    def select_samples(self, locs: Sequence[int]) -> Views:
        """Samples are selected by actual positions, not row names."""
        views_dict = {}
        for k in self.keys():
            views_dict[k] = self.view(key=k).select_rows(selected=locs)
        return JustViews(views_dict=views_dict)

    @abstractmethod
    def n_views(self):
        raise NotImplementedError()

    def serialize(self):
        """Returns views composed of either the same tables or equivalent tables with less memory occupation,
        to optimize message passing or storage space."""
        new_views = {}
        for k in self.keys():
            new_views[k] = self.view(key=k).serialize()
        return JustViews(views_dict=new_views)

    @abstractmethod
    def has_fast_cols(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def fast_cols(self) -> Views:
        raise NotImplementedError()

    def has_view(self, view_name: str) -> bool:
        return view_name in self.keys()

    def select_views(self, view_names: Iterable[str]) -> Views:
        """If a name is present two times, that view will still be added just once."""
        res_dict = {}
        for v in view_names:
            res_dict[v] = self.view(key=v)
        return JustViews(views_dict=res_dict)

    def view_col_numbers(self) -> dict[str, int]:
        return {k: self.view(key=k).n_col() for k in self.keys()}

    def set_view(self, view_name: str, table: Table) -> Views:
        """View will be either added or overwritten."""
        d = self.as_dict()
        d[view_name] = table
        return JustViews(views_dict=d)

    def has_non_finite(self) -> bool:
        for k in self.keys():
            if self.view(key=k).has_non_finite():
                return True
        return False

    @abstractmethod
    def compile(self, max_cells: int = DEFAULT_MAX_CACHEABLE_CELLS) -> Views:
        """All views get compiled, including the collapsed view if they are cached views.
        the max_cells is applied to each table separately."""
        raise NotImplementedError()



class EmptyViews(Views):
    """There are no views but still it has a number of samples and sample labels (These samples have zero views)."""
    __sample_names: FrozenList[str]

    def __init__(self, sample_names: Iterable[str]):
        self.__sample_names = FrozenList(items=sample_names)
        self.__sample_names.freeze()

    def keys(self) -> Set[str]:
        return set()

    def view(self, key: Union[str, int]) -> Table:
        raise ValueError()

    def collapsed(self) -> Table:
        return BackedTable(backend=NpTable(data=np.empty((len(self.__sample_names)), 0),
                                           rownames=self.__sample_names))

    def as_cached(self) -> CachedViews:
        return CachedViews(views=self)

    def n_views(self):
        return 0

    def has_fast_cols(self) -> bool:
        """Having no cols makes them fast."""
        return True

    def fast_cols(self) -> Views:
        return self

    def compile(self, max_cells: int = DEFAULT_MAX_CACHEABLE_CELLS) -> Views:
        return self


class JustViews(Views):
    __views_dict: dict[str, Table]

    def __init__(self, views_dict: dict[str, Table]):
        """Constructor checks if the views have the same number of samples.
        Views can be passed in any order but will be stored sorted by name."""
        self.__views_dict = dict(sorted(views_dict.items()))
        if not self.__n_samples_consistency():
            raise ValueError("Number of samples is not consistent.\n" + str(self))

    def __n_samples_consistency(self) -> bool:
        n = self.n_samples()
        views = self.__views_dict
        for v in views:
            if n_row(views[v]) != n:
                return False
        return True

    def keys(self) -> Set[str]:
        return self.__views_dict.keys()

    def view(self, key: Union[str, int]) -> Table:
        if isinstance(key, str):
            return self.__views_dict[key]
        elif isinstance(key, int):
            d = self.__views_dict
            return list(d.values())[key]
        else:
            raise ValueError()

    def collapsed(self) -> Table:
        """A collapsed view is returned. It is not compiled."""
        return FreshCollapsedTables.create_renamed(tables=list(self.__views_dict.values()))

    def as_cached(self) -> CachedViews:
        return CachedViews(views=self)

    def n_views(self):
        return len(self.__views_dict)

    def has_fast_cols(self) -> bool:
        for v in self.__views_dict.values():
            if not v.has_fast_cols():
                return False
        return True

    def fast_cols(self) -> Views:
        new_views = {}
        for k in self.keys():
            new_views[k] = self.view(key=k).fast_cols()
        return JustViews(views_dict=new_views)

    @staticmethod
    def create_from_dfs(views_dict: dict[str, DataFrame]) -> JustViews:
        """This method does not reset the indices."""
        return JustViews(views_dict={k: BackedTable(backend=NpTable(data=v)) for k, v in views_dict.items()})

    def compile(self, max_cells: int = DEFAULT_MAX_CACHEABLE_CELLS) -> Views:
        views_dict= self.__views_dict
        return JustViews(views_dict={k: v.compile(max_cells=max_cells) for k, v in views_dict.items()})


class CachedViews(Views):
    __inner_views: Views
    __collapsed: Optional[Table]

    def __init__(self, views: Views, collapsed: Optional[Table] = None):
        self.__inner_views = views
        if collapsed is not None:
            if collapsed.n_col() != views.n_features() or collapsed.n_row() != views.n_samples():
                raise ValueError("Passed collapsed table is not of the right shape.")
        self.__collapsed = collapsed

    def keys(self) -> Set[str]:
        return self.__inner_views.keys()

    def view(self, key: Union[str, int]) -> Table:
        return self.__inner_views.view(key=key)

    def collapsed(self) -> Table:
        if self.__collapsed is None:
            self.__collapsed = self.__inner_views.collapsed()
        return self.__collapsed

    def as_cached(self) -> CachedViews:
        return self

    @staticmethod
    def create_from_dict(views_dict: dict[str, Table]) -> CachedViews:
        return CachedViews(JustViews(views_dict=views_dict))

    def n_views(self):
        return self.__inner_views.n_views()

    def fast_cols(self) -> Views:
        # We reuse the collapsed table (making sure it has fast cols) if present.
        if self.has_fast_cols():
            return self
        else:
            collapsed = None
            if self.__collapsed is not None:
                collapsed = self.__collapsed.fast_cols()
            return CachedViews(views=self.__inner_views.fast_cols(), collapsed=collapsed)

    def has_fast_cols(self) -> bool:
        # We do not check if the collapsed table exists and has fast cols.
        # Having fast views and slow collapsed would not make sense,
        # and anyway refreshing it would not make a difference.
        return self.__inner_views.has_fast_cols()

    def compile(self, max_cells: int = DEFAULT_MAX_CACHEABLE_CELLS) -> Views:
        return CachedViews(views=self.__inner_views.compile(max_cells=max_cells),
                           collapsed=self.collapsed().compile(max_cells=max_cells))


EMPTY_VIEWS = CachedViews(JustViews(views_dict={}))