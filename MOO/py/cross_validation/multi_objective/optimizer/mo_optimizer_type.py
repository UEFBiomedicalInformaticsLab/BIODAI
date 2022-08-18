from abc import abstractmethod, ABC

from util.named import NickNamed


class MOOptimizerType(NickNamed, ABC):

    @abstractmethod
    def uses_inner_models(self) -> bool:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name()


class ConcreteMOOptimizerType(MOOptimizerType):
    __uses_inner_models: bool
    __nick: str
    __name: str

    def __init__(self, uses_inner_models: bool, nick: str, name: str = None):
        self.__uses_inner_models = uses_inner_models
        self.__nick = nick
        if name is None:
            self.__name = nick
        else:
            self.__name = name

    def uses_inner_models(self) -> bool:
        return self.__uses_inner_models

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        return self.name()
