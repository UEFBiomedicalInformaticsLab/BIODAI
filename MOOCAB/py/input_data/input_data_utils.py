from collections.abc import Sequence

from input_data.input_data import InputData
from objective.social_objective import SocialObjective


def select_outcomes_in_objectives(input_data: InputData, objectives: Sequence[SocialObjective]) -> InputData:
    outcome_labels = set()
    for o in objectives:
        if o.has_outcome_label():
            outcome_labels.add(o.outcome_label())
    return input_data.select_outcomes(keys=outcome_labels)
