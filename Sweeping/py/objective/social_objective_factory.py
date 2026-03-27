from abc import ABC

from objective.social_objective import SocialObjective
from util.named import NickNamed


class SocialObjectiveFactory(NickNamed, ABC):

    def create(self) -> SocialObjective:
        raise NotImplementedError()

    def name(self) -> str:
        return "objective factory without name"


# The objective does not depend on the other individuals. Always the same objective is returned.
class PersonalObjectiveFactory(SocialObjectiveFactory):

    __objective: SocialObjective

    def __init__(self, objective: SocialObjective):
        self.__objective = objective

    def create(self) -> SocialObjective:
        return self.__objective

    def name(self):
        return self.__objective.name()

    def nick(self) -> str:
        return self.__objective.nick()

    def __str__(self) -> str:
        return "personal objective factory for objective " + str(self.__objective)
