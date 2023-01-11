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

    def eval_invalid(self, pop: Iterable[PeculiarIndividual]):
        """pop is modified in place. Returns individuals that were invalid before the call."""
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
        return invalid_ind

    def n_objectives(self) -> int:
        return len(self.__objectives)

    def n_features(self) -> int:
        return self.__evaluator.n_features()
