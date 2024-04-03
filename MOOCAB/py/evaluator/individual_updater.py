from collections.abc import Sequence
from collections.abc import Iterable

from evaluator.workers_pool_evaluator import WorkersPoolEvaluator
from individual.peculiar_individual import PeculiarIndividual
from objective.social_objective import PersonalObjective


class IndividualUpdater:
    __evaluator: WorkersPoolEvaluator
    __objectives: Sequence[PersonalObjective]

    def __init__(self, evaluator: WorkersPoolEvaluator, objectives: Sequence[PersonalObjective]):
        self.__evaluator = evaluator
        self.__objectives = objectives

    def eval_invalid(self, pop: Iterable[PeculiarIndividual]) -> list[PeculiarIndividual]:
        """pop is modified in place. Returns individuals that were invalid before the call.
        TODO Can use a more specific type of individual, with confidence."""
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
            ind.set_predictors(predictors)
            ind.set_std_dev(std_dev=evals.std_dev())
            ind.set_ci95(ci95=evals.ci95())
            ind.set_bootstrap_mean(bootstrap_mean=evals.bootstrap_mean())
            if evals.has_importances():
                ind.set_personalized_feature_importance(personalized_feature_importance=evals.importances())
        return invalid_ind

    def n_objectives(self) -> int:
        return len(self.__objectives)

    def n_features(self) -> int:
        return self.__evaluator.n_features()
