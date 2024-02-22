from ga_runner.secondary_sorting_strategy.individual_attribute_manager.individual_attribute_manager import \
    IndividualAttributeManager
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from social_space import assign_crowd_dist_all_fronts


class CrowdingDistanceManager(IndividualAttributeManager):
    """Not adding stats for crowding distance since infinities create problems with logbook."""

    def attribute_name(self) -> str:
        return "crowding distance"

    @staticmethod
    def get(ind: PeculiarIndividualByListlike) -> float:
        return ind.get_crowding_distance()

    def compute(self, individuals: list[PeculiarIndividualByListlike], hp_manager: HyperparamManager):
        assign_crowd_dist_all_fronts(individuals=individuals)
