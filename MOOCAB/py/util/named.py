from abc import abstractmethod, ABC


class Named:

    def name(self) -> str:
        return str(self)  # Fallback if name not defined.


class NickNamed(Named, ABC):

    @abstractmethod
    def nick(self) -> str:
        raise NotImplementedError()

    def name(self) -> str:
        return self.nick()
