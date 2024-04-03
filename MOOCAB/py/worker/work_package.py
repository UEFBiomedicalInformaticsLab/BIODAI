from typing import NamedTuple

from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike


class WorkPackage(NamedTuple):
    individual: PeculiarIndividualByListlike

    def __str__(self) -> str:
        res = ""
        res += "Individual:\n"
        res += str(self.individual)
        return res
