from collections.abc import Set
from typing import Optional

from util.named import Named
from util.str_utils import str_dict


class MVFeatureSetByNames(Named):
    __features_by_view: dict[str, set[str]]
    __name: Optional[str]

    def __init__(self, features_by_view: dict[str, set[str]], name: Optional[str] = None):
        self.__features_by_view = features_by_view
        self.__name = name

    def view_names(self) -> Set[str]:
        """Returned object is set-like."""
        return self.__features_by_view.keys()

    def view_features(self, view_name: str) -> set[str]:
        return self.__features_by_view[view_name]

    def __str__(self) -> str:
        return str(self.__features_by_view)

    def name(self):
        if self.__name is None:
            return str_dict(self.__features_by_view)
        else:
            return self.__name

    def n_features(self) -> int:
        res = 0
        for v in self.view_names():
            res += len(self.view_features(view_name=v))
        return res
