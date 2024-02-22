from comparators import Comparator, ComparatorOnCrowdingDistance, ComparatorOnSocialSpace
from ga_runner.secondary_sorting_strategy.individual_attribute_manager.crowding_distance_manager import \
    CrowdingDistanceManager
from ga_runner.secondary_sorting_strategy.individual_attribute_manager.individual_attribute_manager import \
    IndividualAttributeManager
from ga_runner.secondary_sorting_strategy.individual_attribute_manager.peculiarity_manager import PeculiarityManager
from ga_runner.secondary_sorting_strategy.individual_attribute_manager.social_space_manager import SocialSpaceManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from util.named import NickNamed


class SecondarySortingStrategy(NickNamed):
    __attribute_managers: list[IndividualAttributeManager]  # In order of execution
    __secondary_comparator: Comparator
    __name: str
    __nick: str

    def __init__(
            self, attribute_managers: list[IndividualAttributeManager], secondary_comparator: Comparator,
            name: str, nick: str):
        self.__attribute_managers = attribute_managers
        self.__secondary_comparator = secondary_comparator
        self.__name = name
        self.__nick = nick

    def attribute_managers(self) -> list[IndividualAttributeManager]:
        return self.__attribute_managers

    def secondary_comparator(self) -> Comparator:
        return self.__secondary_comparator

    def compute_all_attributes(self, individuals: list[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
        """Calls all attribute managers in order. They compute attributes like crowding distance or social space."""
        for am in self.attribute_managers():
            am.compute(individuals=individuals, hp_manager=hp_manager)

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick


crowding_distance_strategy = SecondarySortingStrategy(
    attribute_managers=[CrowdingDistanceManager()],
    secondary_comparator=ComparatorOnCrowdingDistance(),
    name="Crowding distance strategy",
    nick="Crowd")


social_space_strategy = SecondarySortingStrategy(
    attribute_managers=[CrowdingDistanceManager(), PeculiarityManager(), SocialSpaceManager()],
    secondary_comparator=ComparatorOnSocialSpace(),
    name="Social space strategy",
    nick="Social")
