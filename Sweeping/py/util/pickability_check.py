import pickle
from abc import abstractmethod, ABC

from util.preconditions import check_none
from util.utils import IllegalStateError


class PicklabilityCheckRes(ABC):

    @abstractmethod
    def passed(self) -> bool:
        raise NotImplementedError()

    def dumps_passed(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def exception(self) -> BaseException:
        raise NotImplementedError()


class PassedPicklabilityCheck(PicklabilityCheckRes):

    def passed(self) -> bool:
        return True

    def dumps_passed(self) -> bool:
        return True

    def exception(self) -> BaseException:
        raise IllegalStateError()

    def __str__(self) -> str:
        return "passed"


class FailedPicklabilityCheck(PicklabilityCheckRes, ABC):
    __repeated: bool

    def __init__(self, repeated: bool):
        self.__repeated = repeated

    def repeated(self):
        return self.__repeated


class FailedDumpsPicklabilityCheck(FailedPicklabilityCheck):
    __exception: BaseException

    def __init__(self, exception: BaseException, repeated: bool):
        FailedPicklabilityCheck.__init__(self, repeated=repeated)
        self.__exception = check_none(exception)

    def passed(self) -> bool:
        return False

    def dumps_passed(self) -> bool:
        return False

    def exception(self) -> BaseException:
        return self.__exception

    def __str__(self) -> str:
        res = "failed dumps "
        if self.repeated():
            res += "(repeated)"
        else:
            res += "(not repeated)"
        res += " with exception " + str(self.__exception)
        return res


class FailedLoadsPicklabilityCheck(FailedPicklabilityCheck):
    __exception: BaseException

    def __init__(self, exception: BaseException, repeated: bool):
        FailedPicklabilityCheck.__init__(self, repeated=repeated)
        self.__exception = check_none(exception)

    def passed(self) -> bool:
        return False

    def dumps_passed(self) -> bool:
        return True

    def exception(self) -> BaseException:
        return self.__exception

    def __str__(self) -> str:
        res = "failed loads "
        if self.repeated():
            res += "(repeated)"
        else:
            res += "(not repeated)"
        res += " with exception " + str(self.__exception)
        return res


def picklability_check(obj) -> PicklabilityCheckRes:
    try:
        d = pickle.dumps(obj)
    except BaseException as e:
        try:
            pickle.dumps(obj)
        except BaseException as e:
            return FailedDumpsPicklabilityCheck(e, repeated=True)
        return FailedDumpsPicklabilityCheck(e, repeated=False)
    try:
        pickle.loads(d)
    except BaseException as e:
        try:
            pickle.loads(d)
        except BaseException as e:
            return FailedLoadsPicklabilityCheck(e, repeated=True)
        return FailedLoadsPicklabilityCheck(e, repeated=False)
    return PassedPicklabilityCheck()
