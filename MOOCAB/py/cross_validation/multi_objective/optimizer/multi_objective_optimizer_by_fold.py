from abc import abstractmethod

from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from util.named import NickNamed


class MultiObjectiveOptimizerByFold(NickNamed):

    @abstractmethod
    def optimizer_for_fold(self, fold_index: int) -> MultiObjectiveOptimizer:
        raise NotImplementedError()

    @abstractmethod
    def optimizer_for_all_data(self) -> MultiObjectiveOptimizer:
        raise NotImplementedError()

    @abstractmethod
    def uses_inner_models(self):
        raise NotImplementedError()


class DummyMultiObjectiveOptimizerByFold(MultiObjectiveOptimizerByFold):
    """Always returns the same optimizer."""

    __optimizer: MultiObjectiveOptimizer

    def __init__(self, optimizer: MultiObjectiveOptimizer):
        self.__optimizer = optimizer

    def optimizer_for_fold(self, fold_index: int) -> MultiObjectiveOptimizer:
        return self.__optimizer

    def uses_inner_models(self) -> bool:
        return self.__optimizer.uses_inner_models()

    def nick(self) -> str:
        return self.__optimizer.nick()

    def name(self) -> str:
        return self.__optimizer.name()

    def __str__(self) -> str:
        return str(self.__optimizer)

    def optimizer_for_all_data(self) -> MultiObjectiveOptimizer:
        return self.__optimizer
