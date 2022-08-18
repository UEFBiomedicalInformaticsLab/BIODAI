from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import Optional

from pandas import DataFrame

from multi_view_utils import collapse_views, filter_by_mask
from util.list_like import ListLike


class Views(ABC):

    @abstractmethod
    def keys(self) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def view(self, key: str) -> DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def collapsed(self) -> DataFrame:
        raise NotImplementedError()

    def __getitem__(self, key: str):
        return self.view(key=key)

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


class JustViews(Views):
    __views_dict: dict[str, DataFrame]

    def __init__(self, views_dict: dict[str, DataFrame]):
        self.__views_dict = views_dict

    def keys(self) -> Set[str]:
        return self.__views_dict.keys()

    def view(self, key: str) -> DataFrame:
        return self.__views_dict[key]

    def collapsed(self) -> DataFrame:
        return collapse_views(self.__views_dict)

    def as_cached(self) -> CachedViews:
        return CachedViews(views=self)


class CachedViews(Views):
    __inner_views: Views
    __collapsed: Optional[DataFrame]

    def __init__(self, views: Views):
        self.__inner_views = views
        self.__collapsed = None

    def keys(self) -> Set[str]:
        return self.__inner_views.keys()

    def view(self, key: str) -> DataFrame:
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
