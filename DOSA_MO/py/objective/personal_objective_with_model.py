from abc import ABC
from model.model import ClassModel
from objective.composite_personal_objective import CompositePersonalObjective
from objective.objective_computer import ClassificationObjectiveComputer


class PersonalObjectiveWithModel(CompositePersonalObjective, ABC):

    def __init__(self, objective_computer: ClassificationObjectiveComputer, model: ClassModel, outcome_label: str):
        if model is None or outcome_label is None:
            raise ValueError()
        CompositePersonalObjective.__init__(
            self, objective_computer=objective_computer, model=model, target_label=outcome_label)

    def has_model(self) -> bool:
        return True
