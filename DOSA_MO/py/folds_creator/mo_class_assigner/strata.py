from __future__ import annotations
from collections.abc import Sequence
from copy import copy

from numpy import sort
from pandas import unique

from util.sequence_utils import ordered_counter
from util.utils import sorted_dict, str_dict, dict_sort_by_value


class Stratum:
    __identifier: int
    __name: str

    def __init__(self, identifier: int, name: str):
        self.__identifier = identifier
        self.__name = name

    def identifier(self) -> int:
        return self.__identifier

    def name(self) -> str:
        return self.__name


class Strata:
    __ids: list[int]
    __id_to_names: dict[int, str]

    def __init__(self, ids: Sequence[int], id_to_names: dict[int, str]):
        """Passed sequence and dict safe-copied. Copied dict is sorted by id.
        Ids with 0 occurrences are deleted from id to names dict."""
        self.__ids = list(ids)
        self.__id_to_names = sorted_dict(id_to_names)
        counts = ordered_counter(self.__ids)
        for identifier in self.__id_to_names:
            if identifier not in counts:
                del self.__id_to_names[identifier]

    @staticmethod
    def create_from_names(names: Sequence[str], min_size: int = 1) -> Strata:
        unique_names = sorted(unique(names))
        names_to_id = {}
        id_to_names = {}
        i = 0
        for n in unique_names:
            names_to_id[n] = i
            id_to_names[i] = n
            i += 1
        ids = [names_to_id[n] for n in names]
        return merge_till_needed(strata=Strata(ids=ids, id_to_names=id_to_names), min_size=min_size)

    @staticmethod
    def create_from_ids(ids: Sequence[int], min_size: int = 1) -> Strata:
        id_to_names = {}
        for i in unique(ids):
            id_to_names[i] = str(i)
        return merge_till_needed(strata=Strata(ids=ids, id_to_names=id_to_names), min_size=min_size)

    @staticmethod
    def create_one_stratum(n_samples: int) -> Strata:
        return Strata.create_from_ids(ids=[0]*n_samples)

    def ids(self) -> list[int]:
        return copy(self.__ids)

    def names(self) -> list[str]:
        return [self.__id_to_names[i] for i in self.__ids]

    def counts_dict(self) -> dict[str, int]:
        """Returned dict is sorted by id number. It always returns a new object."""
        id_c = self.id_counts()
        res = {}
        for i in self.__id_to_names:
            res[self.__id_to_names[i]] = id_c[i]
        return res

    def id_counts(self) -> dict[int, int]:
        """Returned dict is sorted by id number. It always returns a new object."""
        return ordered_counter(self.__ids)

    def n_samples(self) -> int:
        return len(self.__ids)

    def n_stratum(self) -> int:
        return len(self.__id_to_names)

    def __str__(self) -> str:
        return "Counts:\n" + str_dict(self.counts_dict(), in_lines=True)

    def id_name(self, identifier: int) -> str:
        return self.__id_to_names[identifier]

    def merge_classes(self, id1: int, id2: int) -> Strata:
        """Merges the classes, not in place."""
        res_id_to_names = copy(self.__id_to_names)
        res_id_to_names[id1] = self.__id_to_names[id1] + " + " + self.__id_to_names[id2]
        del res_id_to_names[id2]
        res_ids = copy(self.__ids)
        for i in range(self.n_samples()):
            if self.__ids[i] == id2:
                res_ids[i] = id1
        return Strata(ids=res_ids, id_to_names=res_id_to_names)


def integrate_strata(strata1: Strata, strata2: Strata, min_size: int = 1) -> Strata:
    n_samples = strata1.n_samples()
    if strata2.n_samples() != n_samples:
        raise ValueError()
    unique_composites = sort(unique(list(zip(strata1.ids(), strata2.ids()))))
    names_map = {}
    mo_map = {}
    next_id = 0
    for c in unique_composites:
        mo_map[c] = next_id
        names_map[next_id] = strata1.id_name(identifier=c[0]) + " " + strata2.id_name(identifier=c[1])
        next_id += 1
    res_ids = []
    for id1, id2 in zip(strata1.ids(), strata2.ids()):
        id_composite = (id1, id2)
        res_ids.append(mo_map[id_composite])
    return merge_till_needed(strata=Strata(ids=res_ids, id_to_names=names_map), min_size=min_size)


def merge_till_needed(strata: Strata, min_size: int) -> Strata:
    proceed = True
    while proceed:
        n_s = strata.n_stratum()
        if n_s > 1:
            counts = dict_sort_by_value(d=strata.id_counts())
            key_list = list(counts.keys())
            size0 = counts[key_list[0]]
            if size0 < min_size:
                strata = strata.merge_classes(id1=key_list[0], id2=key_list[1])
            else:
                proceed = False
        else:
            proceed = False
    return strata
