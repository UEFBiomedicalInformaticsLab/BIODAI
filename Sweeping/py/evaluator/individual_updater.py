from collections.abc import Sequence
from collections.abc import Iterable

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from individual.peculiar_confident_individual import PeculiarConfidentIndividualMV
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance


class IndividualUpdater:
    __evaluator: WorkersPoolEvaluator
    __objectives: Sequence[PersonalObjectiveWithImportance]

    def __init__(self, evaluator: WorkersPoolEvaluator, objectives: Sequence[PersonalObjectiveWithImportance]):
        self.__evaluator = evaluator
        self.__objectives = objectives

    def eval_invalid(self, pop: Iterable[PeculiarConfidentIndividualMV]) -> list[PeculiarConfidentIndividualMV]:
        """pop is modified in place. Returns individuals that were invalid before the call.
        TODO Can use a more specific type of individual, with predictors."""
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        evaluations = self.__evaluator.evaluate_batch(individuals=invalid_ind)
        for i in range(len(invalid_ind)):
            ind = invalid_ind[i]
            evals = evaluations[i]
            fit = evals.fit()
            predictors = evals.predictors()
            if len(fit) == 0:
                raise Exception("Empty fitness!")
            ind.fitness.setValues(values=fit)
            ind.set_std_dev(std_dev=evals.std_dev())
            ind.set_ci95(ci95=evals.ci95())
            ind.set_bootstrap_mean(bootstrap_mean=evals.bootstrap_mean())
            if isinstance(ind, PeculiarIndividualByListlike):
                ind.set_predictors(predictors)
                if evals.has_importances():
                    ind.set_personalized_feature_importance(personalized_feature_importance=evals.importances())
            else:
                raise ValueError()
        return invalid_ind

    def n_objectives(self) -> int:
        return len(self.__objectives)

    def individual_size(self) -> int:
        return self.__evaluator.individual_size()