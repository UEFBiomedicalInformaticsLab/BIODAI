from abc import abstractmethod

from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from util.named import NickNamed


class FeatureImportanceByFold(NickNamed):

    @abstractmethod
    def fi_for_fold(self, fold_index: int) -> MultiViewFeatureImportance:
        raise NotImplementedError()

    @abstractmethod
    def fi_for_all_data(self) -> MultiViewFeatureImportance:
        raise NotImplementedError()


class DummyFeatureImportanceByFold(FeatureImportanceByFold):
    __fi: MultiViewFeatureImportance

    def __init__(self, fi: MultiViewFeatureImportance):
        self.__fi = fi

    def fi_for_fold(self, fold_index: int) -> MultiViewFeatureImportance:
        return self.__fi

    def fi_for_all_data(self) -> MultiViewFeatureImportance:
        return self.__fi

    def nick(self) -> str:
        return self.__fi.nick()

    def name(self) -> str:
        return self.__fi.name()

    def __str__(self) -> str:
        return str(self.__fi)
