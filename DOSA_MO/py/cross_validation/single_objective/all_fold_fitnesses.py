from typing import Optional

from util.hyperbox.hyperbox import Interval
from util.utils import IllegalStateError


class AllFitnesses:
    """Lists with a value for each individual."""
    __test: list[float]
    __test_ci: Optional[list[Interval]]
    __train: list[float]
    __train_ci: Optional[list[Interval]]
    __inner_cv: Optional[list[float]]
    __inner_cv_ci: Optional[list[Interval]]
    __inner_cv_sd: Optional[list[float]]
    __inner_cv_bootstrap_mean: Optional[list[float]]

    def __init__(
            self,
            test: list[float],
            test_ci: Optional[list[Interval]],
            train: list[float],
            train_ci: Optional[list[Interval]],
            inner_cv: Optional[list[float]] = None,
            inner_cv_ci: Optional[list[Interval]] = None,
            inner_cv_sd: Optional[list[float]] = None,
            inner_cv_bootstrap_mean: Optional[list[float]] = None):
        self.__test = test
        self.__test_ci = test_ci
        self.__train = train
        self.__train_ci = train_ci
        self.__inner_cv = inner_cv
        self.__inner_cv_ci = inner_cv_ci
        self.__inner_cv_sd = inner_cv_sd
        self.__inner_cv_bootstrap_mean = inner_cv_bootstrap_mean

    def test(self) -> list[float]:
        return self.__test

    def has_test_ci(self) -> bool:
        return self.__test_ci is not None

    def test_ci(self) -> list[Interval]:
        if self.has_test_ci():
            return self.__test_ci
        else:
            raise IllegalStateError()

    def train(self) -> list[float]:
        return self.__train

    def has_train_ci(self) -> bool:
        return self.__train_ci is not None

    def train_ci(self) -> list[Interval]:
        if self.has_train_ci():
            return self.__train_ci
        else:
            raise IllegalStateError()

    def has_inner_cv(self) -> bool:
        return self.__inner_cv is not None

    def inner_cv(self) -> list[float]:
        if self.has_inner_cv():
            return self.__inner_cv
        else:
            raise IllegalStateError()

    def has_inner_cv_ci(self) -> bool:
        return self.__inner_cv_ci is not None

    def inner_cv_ci(self) -> list[Interval]:
        if self.has_inner_cv_ci():
            return self.__inner_cv_ci
        else:
            raise IllegalStateError()

    def has_inner_cv_sd(self) -> bool:
        return self.__inner_cv_sd is not None

    def inner_cv_sd(self) -> list[float]:
        if self.has_inner_cv_sd():
            return self.__inner_cv_sd
        else:
            raise IllegalStateError()

    def has_inner_cv_bootstrap_mean(self) -> bool:
        return self.__inner_cv_bootstrap_mean is not None

    def inner_cv_bootstrap_mean(self) -> list[float]:
        if self.has_inner_cv_bootstrap_mean():
            return self.__inner_cv_bootstrap_mean
        else:
            raise IllegalStateError()
