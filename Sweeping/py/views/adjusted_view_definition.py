from __future__ import annotations

from collections.abc import Iterable, Sequence

from util.dict_utils import nested_unique_sorted_values
from util.named import NickNamed
from util.str_utils import tuple_to_string


def print_view_and_adjusters(view_name: str, adjusters: set[str], compact: bool) -> str:
    res = ""
    res += str(view_name)
    adjusters = sorted(adjusters)
    if adjusters:
        res += tuple_to_string(adjusters, compact=compact)
    return res


class AdjustedViewDef(NickNamed):
    """The objects are immutable. Views are ordered alphabetically. Views that are not included in this
    definition are considered neither predictive nor adjusting thus not relevant for modelling."""
    __view_to_adjusters: dict[str,set[str]]
    """The keys are the views to be used, the values are the sets of views used to adjust.
    The views are always sorted alphabetically."""

    def __init__(self, view_to_adjusters: dict[str,set[str]]):
        """Views can be passed in any order and will be sorted alphabetically."""
        view_to_adjusters = {k: set(sorted(view_to_adjusters[k])) for k in sorted(view_to_adjusters.keys())}
        all_adjusted_views = view_to_adjusters.keys()
        all_adjusting_views = set()
        for v in view_to_adjusters.values():
            all_adjusting_views.update(v)
        if not all_adjusting_views.isdisjoint(all_adjusted_views):
            raise ValueError("Adjusting views and adjusted views must be disjoint.")
        self.__view_to_adjusters = view_to_adjusters

    def adjusters_for_view(self, view: str) -> set[str]:
        return set(self.__view_to_adjusters[view])

    def needs_adjustment(self) -> bool:
        """True if there is at least one view that needs adjustment."""
        for v in self.__view_to_adjusters.values():
            if v:
                return True
        return False

    def predictive_view_names_set(self) -> set[str]:
        """Names of the views that are directly used for prediction."""
        return set(self.__view_to_adjusters.keys())

    def predictive_view_names_seq(self) -> Sequence[str]:
        """Names of the views that are directly used for prediction. Sorted alphabetically."""
        return list(self.__view_to_adjusters.keys())

    def adjuster_view_names(self) -> list[str]:
        """Names are returned in sorted order."""
        return nested_unique_sorted_values(d=self.__view_to_adjusters)

    def all_views_set(self) -> set[str]:
        return set(self.all_views_seq())

    def all_views_seq(self) -> Sequence[str]:
        res = set()
        res.update(self.predictive_view_names_set())
        res.update(self.adjuster_view_names())
        return sorted(res)

    def make_all_views_predictive(self) -> AdjustedViewDef:
        return AdjustedViewDef.create_unadjusted(view_names=self.all_views_seq())

    @staticmethod
    def create_unadjusted(view_names: Iterable[str]) -> AdjustedViewDef:
        return AdjustedViewDef(view_to_adjusters={name: set() for name in view_names})

    def select_views(self, view_names: Iterable[str]) -> AdjustedViewDef:
        """Only predictive and adjusting views present in the selection are included in the result.
        Trying to select views that are not present does not raise an error."""
        view_names = set(view_names)
        res_dict = {}
        for v in view_names.intersection(self.__view_to_adjusters.keys()):
            res_dict[v] = self.__view_to_adjusters[v].intersection(view_names)
        return AdjustedViewDef(view_to_adjusters=res_dict)

    def __str__(self) -> str:
        return self.__print(sep=", ", compact=False)

    def nick(self) -> str:
        return self.__print(sep="_", compact=True)

    def name(self) -> str:
        return self.__print(sep=" ", compact=True)

    def __print(self, sep: str, compact: bool) -> str:
        view_to_adjusters = self.__view_to_adjusters
        res = ""
        for n in view_to_adjusters:
            if res != "":
                res += sep
            adjusters = view_to_adjusters[n]
            res += print_view_and_adjusters(view_name=n, adjusters=adjusters, compact=compact)
        return res

    def __eq__(self, other) -> bool:
        if not isinstance(other, AdjustedViewDef):
            return False
        return self.__view_to_adjusters == other.__view_to_adjusters

    def __hash__(self) -> int:
        return hash(self.__view_to_adjusters)

    def is_predictive_view(self, view_name: str) -> bool:
        return view_name in self.predictive_view_names_set()

    def is_adjusting_view(self, view_name: str) -> bool:
        return view_name in self. adjuster_view_names()

    def num_predictive_views(self) -> int:
        return len(self.__view_to_adjusters.keys())
