from abc import ABC
from typing import NamedTuple, List

from deap.tools import Logbook, ParetoFront, History

from ga_runner.ga_runner import GARunner
from individual.individual_with_context import IndividualWithContext
from individual.peculiar_individual_with_context import contextualize_all
from util.preconditions import check_none


class ContextualizedGAResult(NamedTuple):
    pop: List[IndividualWithContext]
    logbook: Logbook
    pareto_hof: ParetoFront
    history: History


class ContextualizedGARunner(ABC):

    def run(self, views, outcomes, folds_list, seed=2547, n_workers=1, return_pareto=False, return_history=False,
            initial_pop=None) -> ContextualizedGAResult:
        raise NotImplementedError()


class ContextualizedGARunnerWrapper(ContextualizedGARunner):
    __inner: GARunner

    def __init__(self, inner: GARunner):
        self.__inner = check_none(inner)

    def run(self, views, outcomes, folds_list, seed=2547, n_workers=1, return_pareto=False, return_history=False,
            initial_pop=None) -> ContextualizedGAResult:
        inner_res = self.__inner.run(views=views, outcomes=outcomes, folds_list=folds_list, seed=seed,
                                     n_workers=n_workers,
                                     return_pareto=return_pareto, return_history=return_history,
                                     initial_pop=initial_pop)
        pop = contextualize_all(hps=inner_res.pop, hp_manager=inner_res.hp_manager)
        return ContextualizedGAResult(pop=pop, logbook=inner_res.logbook, pareto_hof=inner_res.pareto_hof,
                                      history=inner_res.history)
