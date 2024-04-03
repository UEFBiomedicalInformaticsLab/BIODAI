from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Set, Sequence
from typing import Optional, Union

from pandas import DataFrame

from multi_view_utils import collapse_views, filter_by_mask
from util.dataframes import n_col, n_row
from util.list_like import ListLike


class Views(ABC):

    @abstractmethod
    def keys(self) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def view(self, key: Union[str, int]) -> DataFrame:
        """If receiving a string it is used as a key, if receiving an int it is used as a position.
        Allows use in zip."""
        raise NotImplementedError()

    @abstractmethod
    def collapsed(self) -> DataFrame:
        raise NotImplementedError()

    def __getitem__(self, key: Union[str, int]):
        """If receiving a string it is used as a key, if receiving an int it is used as a position.
        Allows use in zip."""
        return self.view(key=key)

    def __str__(self) -> str:
        res = ""
        res += "Views (number of columns):\n"
        for vk in self.keys():
            res += vk + " (" + str(n_col(self[vk])) + ")\n"
        res += "Views (features):\n"
        for vk in self.keys():
            res += vk + " (" + str(self[vk].columns) + ")\n"
        return res

    @abstractmethod
    def as_cached(self) -> CachedViews:
        """Cached views cache the collapsed state."""
        raise NotImplementedError()

    def collapsed_filtered_by_mask(self, mask: ListLike) -> DataFrame:
        return filter_by_mask(self.collapsed(), mask)

    def as_dict(self) -> dict[str, DataFrame]:
        res = {}
        for k in self.keys():
            res[k] = self.view(key=k)
        return res

    def n_samples(self) -> int:
        for k in self.keys():
            return n_row(self.view(key=k))
        return 0  # If there are no views.

    def select_samples(self, locs: Sequence[int]) -> Views:
        """Samples are selected by actual positions, not row names."""
        views_dict = {}
        for k in self.keys():
            views_dict[k] = self.view(key=k).take(locs, axis=0)
        return JustViews(views_dict=views_dict)

    @abstractmethod
    def n_views(self):
        raise NotImplementedError()


class JustViews(Views):
    __views_dict: dict[str, DataFrame]

    def __init__(self, views_dict: dict[str, DataFrame]):
        """Constructor checks if the views have the same number of samples."""
        self.__views_dict = views_dict
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

    def view(self, key: Union[str, int]) -> DataFrame:
        if isinstance(key, str):
            return self.__views_dict[key]
        elif isinstance(key, int):
            d = self.__views_dict
            return list(d.values())[key]

    def collapsed(self) -> DataFrame:
        return collapse_views(self.__views_dict)

    def as_cached(self) -> CachedViews:
        return CachedViews(views=self)

    def n_views(self):
        return len(self.__views_dict)


class CachedViews(Views):
    __inner_views: Views
    __collapsed: Optional[DataFrame]

    def __init__(self, views: Views):
        self.__inner_views = views
        self.__collapsed = None

    def keys(self) -> Set[str]:
        return self.__inner_views.keys()

    def view(self, key: Union[str, int]) -> DataFrame:
        return self.__inner_views.view(key=key)

    def collapsed(self) -> DataFrame:
        if self.__collapsed is None:
            self.__collapsed = self.__inner_views.collapsed()
        return self.__collapsed

    def as_cached(self) -> CachedViews:
        return self

    @staticmethod
    def create_from_dict(views_dict: dict[str, DataFrame]) -> CachedViews:
        return CachedViews(JustViews(views_dict=views_dict))

    def n_views(self):
        return self.__inner_views.n_views()
