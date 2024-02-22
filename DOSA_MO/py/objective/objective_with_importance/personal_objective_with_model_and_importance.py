from abc import ABC
from model.model import ClassModel
from objective.objective_with_importance.personal_objective_with_importance import \
    CompositePersonalObjectiveWithImportance
from objective.objective_with_importance.objective_computer_with_importance import \
    ClassificationObjectiveComputerWithImportance


class PersonalObjectiveWithModelAndImportance(CompositePersonalObjectiveWithImportance, ABC):

    def __init__(
            self, objective_computer: ClassificationObjectiveComputerWithImportance,
            model: ClassModel, outcome_label: str):
        if model is None or outcome_label is None:
            raise ValueError()
        CompositePersonalObjectiveWithImportance.__init__(
            self, objective_computer=objective_computer, model=model, target_label=outcome_label)

    def has_model(self) -> bool:
        return True
