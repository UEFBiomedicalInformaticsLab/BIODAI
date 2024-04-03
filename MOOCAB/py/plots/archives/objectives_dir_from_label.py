from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from model.model import Model
from objective.composite_personal_objective import CompositePersonalObjective
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import Leanness, RootLeanness
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from objective.social_objective import PersonalObjective
from plots.plot_labels import RF_LAB, RF_LEGACY_LAB
from setup.parse_objectives import parse_model
from setup.setup_utils import combine_objective_strings


def default_objective_from_computer_and_model(
        objective_computer: ObjectiveComputer, model: Model = None) -> PersonalObjective:
    if model is not None and objective_computer.requires_target():
        return CompositePersonalObjective(
            objective_computer=objective_computer, model=model, target_label="target")
    else:
        return CompositePersonalObjective(objective_computer=objective_computer)


def default_objective_from_computer_and_model_lab(
        objective_computer: ObjectiveComputer, model_lab: str = None) -> PersonalObjective:
    if model_lab is not None:
        model = parse_model(model_lab)
    else:
        model = None
    return default_objective_from_computer_and_model(objective_computer=objective_computer, model=model)


class ObjectivesDirFromLabel(ABC):

    @abstractmethod
    def objectives_dir_from_label(
            self, classification_inner_lab: Optional[str], survival_inner_lab: Optional[str] = None) -> str:
        """Returns the string composed by the objectives."""
        raise NotImplementedError()

    @abstractmethod
    def has_classification(self) -> bool:
        """Returns true if there is at least a classification objective."""
        raise NotImplementedError()


class ObjectivesDirFromLabelByComputers(ObjectivesDirFromLabel):
    __objectives: Sequence[ObjectiveComputer]

    def __init__(self, objectives: Sequence[ObjectiveComputer]):
        self.__objectives = objectives

    def objectives_dir_from_label(
            self, classification_inner_lab: Optional[str] = None, survival_inner_lab: Optional[str] = None) -> str:
        objective_strings = []
        for o in self.__objectives:
            if o.is_class_objective_computer():
                if classification_inner_lab == RF_LEGACY_LAB and o.requires_predictions():
                    obj_nick = RF_LAB + "_" + o.nick()
                else:
                    obj_nick = default_objective_from_computer_and_model_lab(
                        objective_computer=o, model_lab=classification_inner_lab).nick()
            elif o.is_survival_objective_computer():
                obj_nick = default_objective_from_computer_and_model_lab(
                    objective_computer=o, model_lab=survival_inner_lab).nick()
            else:
                obj_nick = default_objective_from_computer_and_model_lab(objective_computer=o).nick()
            objective_strings.append(obj_nick)
        return combine_objective_strings(objective_strings=objective_strings)

    def has_classification(self) -> bool:
        """Returns true if there is at least a classification objective."""
        for o in self.__objectives:
            if o.is_class_objective_computer():
                return True
        return False


class BalAccLeanness(ObjectivesDirFromLabelByComputers):

    def __init__(self):
        ObjectivesDirFromLabelByComputers.__init__(self, objectives=[BalancedAccuracy(), Leanness()])


class BalAccRootLeannessCIndex(ObjectivesDirFromLabelByComputers):

    def __init__(self):
        ObjectivesDirFromLabelByComputers.__init__(self, objectives=[BalancedAccuracy(), RootLeanness(), CIndex()])
