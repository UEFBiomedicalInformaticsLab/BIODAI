from collections.abc import Sequence, Iterable

from cross_validation.multi_objective.optimizer.user_provided.known_biomarkers import PAM50_NAME, PAM50
from cross_validation.multi_objective.optimizer.user_provided.user_provided_mo_optimizer import UserProvidedMoOptimizer
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from objective.social_objective import PersonalObjective


class Pam50(UserProvidedMoOptimizer):

    def __init__(self,
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 folds_creator: InputDataFoldsCreator,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),)):
        UserProvidedMoOptimizer.__init__(self=self,
                                         feature_sets=[PAM50],
                                         objectives=objectives,
                                         folds_creator=folds_creator,
                                         hof_factories=hof_factories,
                                         optimizer_type_nick=PAM50_NAME,
                                         optimizer_type_name=PAM50_NAME)
