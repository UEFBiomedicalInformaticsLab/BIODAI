from ga_runner.secondary_sorting_strategy.individual_attribute_manager.individual_attribute_manager import \
    IndividualAttributeManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from social_space import assign_just_social_space


class SocialSpaceManager(IndividualAttributeManager):

    def attribute_name(self) -> str:
        return "social space"

    @staticmethod
    def get(ind: PeculiarIndividualByListlike) -> float:
        return ind.get_social_space()

    def compute(self, individuals: list[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
        assign_just_social_space(individuals=individuals)

    @staticmethod
    def add_to_stats() -> bool:
        return True
