from typing import Any

from util.backed_dict.backed_dict import BackedDict
from util.named import NickNamed


class BackedDictProperty(NickNamed):
    __property_name: str
    __name_for_humans: str
    __nick_for_humans: str

    def __init__(self, property_name: str, name_for_humans: str, nick_for_humans: str):
        self.__property_name = property_name
        self.__name_for_humans = name_for_humans
        self.__nick_for_humans = nick_for_humans

    def property_name(self) -> str:
        return self.__property_name

    def name(self) -> str:
        """Name for humans."""
        return self.__name_for_humans

    def nick(self) -> str:
        return self.__nick_for_humans

    def __str__(self) -> str:
        return self.name()
    
    def extract(self, backed_dict: BackedDict) -> Any:
        return backed_dict.get(property_name=self.property_name())
