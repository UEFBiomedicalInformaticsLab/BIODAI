from collections.abc import Iterable
from typing import NamedTuple

from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from objective.social_objective import PersonalObjective
from util.sequence_utils import str_in_lines


class WorkPackage(NamedTuple):
    individual: PeculiarIndividualByListlike
    objectives: Iterable[PersonalObjective]  # TODO We can pass this once in initialization of worker.

    def __str__(self) -> str:
        res = ""
        res += "Individual:\n"
        res += str(self.individual)
        res += "Objectives:\n"
        res += str_in_lines(self.objectives)
        return res
