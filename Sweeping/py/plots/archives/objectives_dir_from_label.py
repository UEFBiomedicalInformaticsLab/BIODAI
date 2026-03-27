from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from model.multi_view.multi_view_model import MVModel
from objective.composite_personal_objective import CompositePersonalObjective
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import Leanness, RootLeanness
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from objective.social_objective import PersonalObjective
from plots.plot_labels import RF_LAB, RF_LEGACY_LAB
from setup.parse_objectives import parse_model_mv
from setup.setup_utils import combine_objective_strings


def default_objective_from_computer_and_model(
        objective_computer: ObjectiveComputer, model: MVModel = None) -> PersonalObjective:
    if model is not None and objective_computer.requires_target():
        return CompositePersonalObjective(
            objective_computer=objective_computer, model=model, target_label="target")
    else:
        return CompositePersonalObjective(objective_computer=objective_computer)


def default_objective_from_computer_and_model_lab(
        objective_computer: ObjectiveComputer, model_lab: str = None) -> PersonalObjective:
    if model_lab is not None:
        model = parse_model_mv(model_lab, objective_computer=objective_computer)
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


RF_LAB_ = RF_LAB + "_"


class ObjectivesDirFromLabelByComputers(ObjectivesDirFromLabel):
    __objectives: Sequence[ObjectiveComputer]
    __non_predictive_nicks: Sequence[str]

    def __init__(self, objectives: Sequence[ObjectiveComputer]):
        self.__objectives = objectives
        self.__non_predictive_nicks = [
            default_objective_from_computer_and_model_lab(objective_computer=o).nick() for o in objectives]

    def objectives_dir_from_label(
            self, classification_inner_lab: Optional[str] = None, survival_inner_lab: Optional[str] = None) -> str:
        objective_strings = []
        for i, o in enumerate(self.__objectives):
            if o.is_class_objective_computer():
                if classification_inner_lab == RF_LEGACY_LAB and o.requires_predictions():
                    obj_nick = RF_LAB_ + o.nick()
                else:
                    obj_nick = default_objective_from_computer_and_model_lab(
                        objective_computer=o, model_lab=classification_inner_lab).nick()
            elif o.is_survival_objective_computer():
                obj_nick = default_objective_from_computer_and_model_lab(
                    objective_computer=o, model_lab=survival_inner_lab).nick()
            else:
                obj_nick = self.__non_predictive_nicks[i]
            objective_strings.append(obj_nick)
        return combine_objective_strings(objective_strings=objective_strings)

    def has_classification(self) -> bool:
        """Returns true if there is at least a classification objective."""
        for o in self.__objectives:
            if o.is_class_objective_computer():
                return True
        return False

    def __str__(self) -> str:
        return "ObjectivesDirFromLabelByComputers " + str(self.__objectives)


class BalAccLeanness(ObjectivesDirFromLabelByComputers):

    def __init__(self):
        ObjectivesDirFromLabelByComputers.__init__(self, objectives=[BalancedAccuracy(), Leanness()])


class BalAccRootLeannessCIndex(ObjectivesDirFromLabelByComputers):

    def __init__(self):
        ObjectivesDirFromLabelByComputers.__init__(self, objectives=[BalancedAccuracy(), RootLeanness(), CIndex()])
