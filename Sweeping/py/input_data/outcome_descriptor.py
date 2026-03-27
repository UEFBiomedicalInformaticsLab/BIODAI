from abc import abstractmethod, ABC

from input_data.outcome_type import OutcomeType
from util.named import Named
from util.utils import IllegalStateError


class OutcomeDescriptor(Named, ABC):
    __name: str

    def __init__(self, name: str):
        self.__name = name

    def name(self) -> str:
        return self.__name

    @abstractmethod
    def outcome_type(self) -> OutcomeType:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name() + " (" + self.outcome_type().name + ")"


class OutcomeDescriptorCategorical(OutcomeDescriptor):

    def __init__(self, name: str):
        OutcomeDescriptor.__init__(self=self, name=name)

    def outcome_type(self) -> OutcomeType:
        return OutcomeType.categorical


class OutcomeDescriptorSurvival(OutcomeDescriptor):

    def __init__(self, name: str):
        OutcomeDescriptor.__init__(self=self, name=name)

    def outcome_type(self) -> OutcomeType:
        return OutcomeType.survival


class OutcomeDescriptorWithColumns(OutcomeDescriptor, ABC):

    def __init__(self, name: str):
        OutcomeDescriptor.__init__(self=self, name=name)

    @abstractmethod
    def categories_col(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def event_col(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def time_col(self) -> str:
        raise NotImplementedError()


class OutcomeDescriptorWithColumnsCategorical(OutcomeDescriptorWithColumns, OutcomeDescriptorCategorical):
    __categories_col: str  # The column in the csv

    def __init__(self, name: str, categories_col: str):
        OutcomeDescriptorWithColumns.__init__(self=self, name=name)
        self.__categories_col = categories_col

    def categories_col(self) -> str:
        return self.__categories_col

    def event_col(self) -> str:
        raise IllegalStateError()

    def time_col(self) -> str:
        raise IllegalStateError()


class OutcomeDescriptorWithColumnsSurvival(OutcomeDescriptorWithColumns, OutcomeDescriptorSurvival):
    __event_col: str
    __time_col: str

    def __init__(self, name: str, event_col: str, time_col: str):
        OutcomeDescriptorWithColumns.__init__(self=self, name=name)
        self.__event_col = event_col
        self.__time_col = time_col

    def categories_col(self) -> str:
        raise IllegalStateError()

    def event_col(self) -> str:
        return self.__event_col

    def time_col(self) -> str:
        return self.__time_col
