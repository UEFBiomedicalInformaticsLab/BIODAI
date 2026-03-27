from random import random
from collections.abc import Iterable, Iterator

from util.utils import bound


def particle_update(x: Iterable[float], v: Iterable[float],
                    p_best: Iterable[float], g_best: Iterable[float],
                    w: float, c1: float, c2: float, r1: float, r2: float, v_max: float) -> Iterator[tuple[float, float]]:
    """Returns an iterator of tuples (position, velocity)"""
    p_mult = c1 * r1
    g_mult = c2 * r2
    neg_v_max = -v_max
    for xd, vd, pd, gd in zip(x, v, p_best, g_best):
        vd_next = w * vd + p_mult * (pd - xd) + g_mult * (gd - xd)
        vd_next = bound(x=vd_next, min_x=neg_v_max, max_x=v_max)
        xd_next = xd + vd_next
        yield xd_next, vd_next


class ParticleUpdater:
    __w: float
    __c1: float
    __c2: float
    __v_max: float

    def __init__(self, w: float, c1: float, c2: float, v_max: float):
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2
        self.__v_max = v_max

    def update_particle(self, x: Iterable[float], v: Iterable[float],
                    p_best: Iterable[float], g_best: Iterable[float]) -> Iterator[tuple[float, float]]:
        """Returns an iterator of tuples (position, velocity)"""
        return particle_update(
            x=x, v=v, p_best=p_best, g_best=g_best, w=self.__w, c1=self.__c1, c2=self.__c2,
            r1=random(), r2=random(), v_max=self.__v_max)
