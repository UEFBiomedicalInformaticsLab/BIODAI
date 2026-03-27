from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV
from individual.individual_with_context_mv import IndividualWithContextMV
from individual.peculiar_individual_with_context import PeculiarIndividualWithContext


class PeculiarIndividualWithContextMV(PeculiarIndividualWithContext, IndividualWithContextMV):
    """Should be treated as unmodifiable, otherwise behavior is unspecified."""

    def __init__(self, individual: ConfidentPredictiveIndividualMV, hp_manager: MvHyperparamManager):
        PeculiarIndividualWithContext.__init__(self=self, individual=individual)
        IndividualWithContextMV.__init__(self=self, individual=individual, hp_manager=hp_manager)
