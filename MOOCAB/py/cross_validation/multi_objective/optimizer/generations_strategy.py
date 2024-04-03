from collections.abc import Sequence
from copy import copy

from util.named import NickNamed
from util.sequence_utils import sequence_to_string


class GenerationsStrategy(NickNamed):
    __sweeps: list[int]
    __concatenated: int

    def __init__(self, sweeps: Sequence[int] = (), concatenated: int = 0):
        clean_sweeps = []
        for s in sweeps:
            if s > 0:
                clean_sweeps.append(s)
        self.__sweeps = clean_sweeps
        self.__concatenated = max(concatenated, 0)

    def num_sweeps(self):
        return len(self.__sweeps)

    def sweep_generations(self, sweep_number: int) -> int:
        """Sweep number starts from 0"""
        return self.__sweeps[sweep_number]

    def concatenated_generations(self) -> int:
        return self.__concatenated

    def nick(self) -> str:
        res = ""
        if self.num_sweeps() > 0:
            res += sequence_to_string(self.__sweeps, compact=True)
        if self.concatenated_generations() > 0:
            res += str(self.concatenated_generations())
        return res

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        res = ""
        if self.num_sweeps() > 0:
            res += sequence_to_string(self.__sweeps)
        if self.concatenated_generations() > 0:
            if res != "":
                res += " "
            res += str(self.concatenated_generations())
        return res

    def sweeping_list(self) -> list[int]:
        return copy(self.__sweeps)
