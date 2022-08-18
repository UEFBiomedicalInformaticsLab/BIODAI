from abc import abstractmethod, ABC

from util.named import NickNamed


class FoldsCreator(NickNamed, ABC):

    @abstractmethod
    def create_folds(self, x, y, seed: int):
        """ Creates the folds used the passed random seed.
            There are no guarantees on the state of the builtin random generators after this call,
            since different instances may use different generators.
            """
        raise NotImplementedError()
