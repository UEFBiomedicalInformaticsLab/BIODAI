from collections.abc import Sequence

from individual.fit import Fit
from util.named import NickNamed


class Hofers(Sequence[Fit], NickNamed):
    __list: list[Fit]
    __name: str
    __nick: str

    def __init__(self, elems: list[Fit], name: str, nick: str):
        self.__list = elems
        self.__name = name
        self.__nick = nick

    def __getitem__(self, i: int) -> Fit:
        return self.__list[i]

    def __len__(self) -> int:
        return len(self.__list)

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick
