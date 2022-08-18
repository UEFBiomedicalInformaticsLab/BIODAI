from collections.abc import Iterable

from input_data.input_data import InputData
from objective.social_objective import PersonalObjective


def select_outcomes_in_objectives(input_data: InputData, objectives: Iterable[PersonalObjective]) -> InputData:
    outcome_labels = set()
    for o in objectives:
        if o.has_outcome_label():
            outcome_labels.add(o.outcome_label())
    return input_data.select_outcomes(keys=outcome_labels)
