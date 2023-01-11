from typing import Optional

from model.model import Model
from objective.objective_computer import Leanness, ObjectiveComputer
from objective.social_objective import PersonalObjective
from util.utils import IllegalStateError


class CompositePersonalObjective(PersonalObjective):

    __objective_computer: ObjectiveComputer
    __model: Optional[Model]
    __target_label: Optional[str]

    def __init__(self, objective_computer: ObjectiveComputer, model: Model = None, target_label: str = None):
        self.__objective_computer = objective_computer
        if objective_computer.requires_predictions():
            if target_label is None and model is not None:
                raise ValueError("If there is a model there must be also a target label.")
            self.__model = model
            self.__target_label = target_label
        else:
            self.__model = None
            self.__target_label = None

    def objective_computer(self) -> ObjectiveComputer:
        return self.__objective_computer

    def has_model(self) -> bool:
        return self.__model is not None

    def has_outcome_label(self) -> bool:
        return self.__target_label is not None

    def model(self) -> Model:
        if self.has_model():
            return self.__model
        else:
            raise IllegalStateError(str(self))

    def outcome_label(self) -> str:
        """An objective can have no model (e.g. if a single independent model is used for all the objectives)
        but still have an outcome."""
        if self.has_outcome_label():
            return self.__target_label
        else:
            raise IllegalStateError(str(self))


class LeannessObjective(CompositePersonalObjective):

    def __init__(self):
        CompositePersonalObjective.__init__(self, objective_computer=Leanness())
