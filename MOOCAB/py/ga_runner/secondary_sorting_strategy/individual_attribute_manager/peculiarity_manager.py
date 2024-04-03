from ga_runner.secondary_sorting_strategy.individual_attribute_manager.individual_attribute_manager import \
    IndividualAttributeManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from social_space import assign_peculiarity


class PeculiarityManager(IndividualAttributeManager):

    def attribute_name(self) -> str:
        return "peculiarity"

    @staticmethod
    def get(ind: PeculiarIndividualByListlike) -> float:
        return ind.get_peculiarity()

    def compute(self, individuals: list[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
        assign_peculiarity(individuals=individuals, hp_manager=hp_manager)

    @staticmethod
    def add_to_stats() -> bool:
        return True
