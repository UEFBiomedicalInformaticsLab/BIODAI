from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Any

from scipy.stats import pearsonr, spearmanr, kendalltau
from math import nan

from util.named import NickNamed


class SequencesToFloat(NickNamed, ABC):
    __error_as_nan: bool

    def __init__(self, error_as_nan: bool = True):
        self.__error_as_nan = error_as_nan

    @abstractmethod
    def apply(self, seq1: Sequence[float], seq2: Sequence[float]) -> float:
        raise NotImplementedError()

    def error_as_nan(self) -> bool:
        return self.__error_as_nan


class Correlation(SequencesToFloat):
    __correlation_function: Any
    __name: str
    __nick: str
    __p_val: bool

    def __init__(self, correlation_function, function_name: str,
                 function_nick: str, p_val: bool = False,
                 error_as_nan: bool = True):
        SequencesToFloat.__init__(self=self, error_as_nan=error_as_nan)
        self.__correlation_function = correlation_function
        self.__p_val = p_val
        self.__nick = function_nick
        self.__name = function_name
        if p_val:
            self.__nick += "_p_val"
            self.__name += " p-value"

    def apply(self, seq1: Sequence[float], seq2: Sequence[float]) -> float:
        try:
            corr_res = self.__correlation_function(seq1, seq2)
        except ValueError as e:
            if self.error_as_nan():
                return nan
            else:
                raise ValueError(
                    str(e) + "\n" +
                    "seq1: " + str(seq1) + "\n" +
                    "seq2: " + str(seq2) + "\n")
        if self.__p_val:
            return corr_res[1]
        else:
            return corr_res[0]

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick


class PearsonCorr(Correlation):

    def __init__(self, p_val: bool = False):
        super().__init__(
            correlation_function=pearsonr,
            function_name="Pearson correlation coefficient",
            function_nick="Pearson",
            p_val=p_val)


class SpearmanCorr(Correlation):

    def __init__(self, p_val: bool = False):
        super().__init__(
            correlation_function=spearmanr,
            function_name="Spearman rank correlation coefficient",
            function_nick="Spearman",
            p_val=p_val)


class KendallCorr(Correlation):

    def __init__(self, p_val: bool = False):
        super().__init__(
            correlation_function=kendalltau,
            function_name="Kendall rank correlation coefficient",
            function_nick="Kendall",
            p_val=p_val)
