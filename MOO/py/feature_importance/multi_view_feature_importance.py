from abc import ABC, abstractmethod

from feature_importance.feature_importance import FeatureImportance
from feature_importance.feature_importance_by_lasso import FeatureImportanceByLasso
from feature_importance.feature_importance_uniform import FeatureImportanceUniform
from input_data.input_data import InputData
from util.distribution.distribution import Distribution
from util.named import NickNamed
from util.utils import IllegalStateError


class MultiViewFeatureImportance(NickNamed, ABC):
    """Works with InputData containing multiple views but just one outcome."""

    @abstractmethod
    def compute(self, input_data: InputData, n_proc: int = 1) -> list[Distribution]:
        raise NotImplementedError()

    @abstractmethod
    def is_none(self) -> bool:
        raise NotImplementedError()


class ConcreteMultiViewFeatureImportance(MultiViewFeatureImportance):
    __fi: FeatureImportance

    def __init__(self, feature_importance: FeatureImportance):
        self.__fi = feature_importance

    def compute(self, input_data: InputData, n_proc: int = 1) -> list[Distribution]:
        if input_data.n_outcomes() != 1:
            raise ValueError("Works with InputData containing multiple views but just one outcome.")
        views = input_data.views()
        y = input_data.outcomes()[0].data()
        return [self.__fi.compute(x=views[v], y=y, n_proc=n_proc) for v in views]

    def is_none(self) -> bool:
        return False

    def name(self) -> str:
        return "multi-view " + self.__fi.name()

    def nick(self) -> str:
        return "MV_" + self.__fi.nick()

    def __str__(self) -> str:
        res = "multi-view feature importance with "
        res += str(self.__fi)
        return res


class MVFeatureImportanceNone(MultiViewFeatureImportance):

    def compute(self, input_data: InputData, n_proc: int = 1) -> list[Distribution]:
        raise IllegalStateError()

    def is_none(self) -> bool:
        return True

    def nick(self) -> str:
        return "none"

    def name(self) -> str:
        return "multi-view none"

    def __str__(self) -> str:
        return "multi-view feature importance none"


class MVFeatureImportanceUniform(ConcreteMultiViewFeatureImportance):

    def __init__(self):
        ConcreteMultiViewFeatureImportance.__init__(self, feature_importance=FeatureImportanceUniform())

    def __str__(self) -> str:
        return "multi-view feature importance uniform"


class MVFeatureImportanceLasso(ConcreteMultiViewFeatureImportance):

    def __init__(self):
        ConcreteMultiViewFeatureImportance.__init__(self, feature_importance=FeatureImportanceByLasso())

    def __str__(self) -> str:
        return "multi-view feature importance lasso"
