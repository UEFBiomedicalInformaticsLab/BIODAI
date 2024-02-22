from __future__ import annotations

from abc import ABC
from typing import Optional

from model.model import Model
from objective.objective_with_importance.objective_computer_with_importance import ObjectiveComputerWithImportance
from objective.objective_with_importance.social_objective_with_importance import SocialObjectiveWithImportance, \
    CompositeSocialObjectiveWithImportance
from objective.social_objective import PersonalObjective


class PersonalObjectiveWithImportance(PersonalObjective, SocialObjectiveWithImportance, ABC):

    def change_computer(self, objective_computer: ObjectiveComputerWithImportance) -> PersonalObjectiveWithImportance:
        """Returns a new instance."""
        model = None
        if self.has_model():
            model = self.model()
        outcome_label = None
        if self.has_outcome_label():
            outcome_label = self.outcome_label()
        return CompositePersonalObjectiveWithImportance(
            objective_computer=objective_computer,
            model=model,
            target_label=outcome_label)


class CompositePersonalObjectiveWithImportance(CompositeSocialObjectiveWithImportance, PersonalObjectiveWithImportance):

    __objective_computer: ObjectiveComputerWithImportance
    __model: Optional[Model]
    __target_label: Optional[str]

    def __init__(self, objective_computer: ObjectiveComputerWithImportance,
                 model: Optional[Model] = None, target_label: Optional[str] = None):
        CompositeSocialObjectiveWithImportance.__init__(
            self=self,
            objective_computer=objective_computer,
            model=model,
            target_label=target_label
        )
