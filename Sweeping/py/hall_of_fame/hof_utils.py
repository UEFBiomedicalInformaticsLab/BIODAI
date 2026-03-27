from copy import deepcopy
from typing import Iterable

from hall_of_fame.hofers import Hofers
from individual.fit import Fit


def hof_path(optimizer_save_path: str, hof_nick: str) -> str:
    return optimizer_save_path + "hofs/" + hof_nick + "/"


def iterable_to_hofers(elems: Iterable[Fit], name: str, nick: str) -> Hofers:
    """Elements are deepcopied."""
    res = []
    for h in elems:
        res.append(deepcopy(h))
    res.sort(key=lambda e: e.get_test_fitness(), reverse=True)
    return Hofers(elems=res, name=name, nick=nick)
