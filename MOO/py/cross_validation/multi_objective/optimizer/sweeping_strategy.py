from collections.abc import Sequence
from util.named import NickNamed
from util.sequence_utils import sequence_to_string


class SweepingStrategy(NickNamed, Sequence):
    __sweeps: [int]

    def __init__(self, sweeps: [int]):
        self.__sweeps = sweeps

    def num_sweeps(self):
        return len(self.__sweeps)

    def generations(self, sweep_number: int) -> int:
        """Sweep number starts from 0"""
        return self.__sweeps[sweep_number]

    def nick(self) -> str:
        return sequence_to_string(self.__sweeps, compact=True)

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return sequence_to_string(self.__sweeps)

    def __getitem__(self, i: int) -> int:
        return self.__sweeps[i]

    def __len__(self) -> int:
        return len(self.__sweeps)
