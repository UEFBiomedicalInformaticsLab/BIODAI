from __future__ import annotations

from typing import Sequence

from pandas import DataFrame

from individual.individual_with_context import IndividualWithContext
from model.mv_predictor import MVPredictor
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.hyperbox.hyperbox import Hyperbox0B
from util.named import NickNamed
from util.sequence_utils import sequence_to_string
from util.utils import name_value


class MultiObjectiveOptimizerResult(NickNamed):
    """One hall of fame on one fold.
    For each individual there is a list of predictors, one for each objective."""
    __name: str
    __nick: str
    __predictors: Sequence[Sequence[MVPredictor]]
    __hyperparams: Sequence[IndividualWithContext]

    def __init__(self, name: str, nick: str, predictors: Sequence[Sequence[MVPredictor]],
                 hyperparams: Sequence[IndividualWithContext]):
        if len(predictors) != len(hyperparams):
            raise ValueError()
        self.__name = name
        self.__nick = nick
        self.__predictors = predictors
        self.__hyperparams = hyperparams

    def predictors(self) -> Sequence[Sequence[MVPredictor]]:
        return self.__predictors

    def predictors_for_objective(self, objective_num: int) -> list[MVPredictor]:
        return [i[objective_num] for i in self.__predictors]

    def hyperparams(self) -> Sequence[IndividualWithContext]:
        return self.__hyperparams

    def fitness_hyperboxes(self) -> Sequence[Hyperbox0B]:
        return [h.fitness_hyperbox() for h in self.__hyperparams]

    def __str__(self):
        res = ""
        res += name_value("Name", self.name()) + "\n"
        res += name_value("Nick", self.nick()) + "\n"
        n_solutions = len(self.__predictors)
        res += "Number of solutions: " + str(n_solutions) + "\n"
        if n_solutions <= 10:
            for i in range(n_solutions):
                res += "hyperparams: " + self.__hyperparams[i].brief_str() + "\n"
                res += "predictor: " + str(self.__predictors[i]) + "\n"
        elif n_solutions <= 20:
            for i in range(n_solutions):
                res += self.__hyperparams[i].brief_str() + "\n"
        return res

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick

    def individuals_to_df(self) -> DataFrame:
        """A df with a row for each individual (keeping order) and a column for each feature.
        0 or 1 for absence/presence. All columns are kept."""
        if len(self.__hyperparams) == 0:
            return DataFrame()
        return DataFrame(data=[h.active_features_mask() for h in self.__hyperparams])

    def has_fitnesses(self) -> bool:
        for h in self.__hyperparams:
            if not h.has_fitness():
                return False
        return True

    def fitnesses_to_df(self) -> DataFrame:
        if len(self.__hyperparams) == 0:
            return DataFrame()
        if self.has_fitnesses():
            return DataFrame(data=[h.fitness.as_list() for h in self.__hyperparams])
        else:
            return DataFrame()

    def std_dev_to_df(self) -> DataFrame:
        if len(self.__hyperparams) == 0:
            return DataFrame()
        if self.has_fitnesses():
            return DataFrame(data=[list(h.std_dev()) for h in self.__hyperparams])
        else:
            return DataFrame()

    def ci_min_to_df(self) -> DataFrame:
        if len(self.__hyperparams) == 0:
            return DataFrame()
        if self.has_fitnesses():
            return DataFrame(data=[(None if c is None else c.a() for c in h.ci95()) for h in self.__hyperparams])
        else:
            return DataFrame()

    def ci_max_to_df(self) -> DataFrame:
        if len(self.__hyperparams) == 0:
            return DataFrame()
        if self.has_fitnesses():
            return DataFrame(data=[(None if c is None else c.b() for c in h.ci95()) for h in self.__hyperparams])
        else:
            return DataFrame()

    def downlift(self, lifter: FeatureSpaceLifterMV) -> MultiObjectiveOptimizerResult:
        predictors = []
        for s in self.__predictors:
            s_pred = []
            for p in s:
                if p is None:
                    s_pred.append(None)
                else:
                    s_pred.append(p.downlift(lifter))
            predictors.append(s_pred)
        return MultiObjectiveOptimizerResult(
            name=self.__name,
            nick=self.__nick,
            predictors=predictors,
            hyperparams=[i.downlift(lifter) for i in self.__hyperparams]
        )

    def select_individuals(self, individuals=Sequence[IndividualWithContext]) -> MultiObjectiveOptimizerResult:
        res_h = []
        res_p = []
        for i, h in enumerate(self.__hyperparams):
            if h in individuals:
                res_h.append(h)
                res_p.append(self.__predictors[i])
        return MultiObjectiveOptimizerResult(
            name=self.__name,
            nick=self.__nick,
            predictors=res_p,
            hyperparams=res_h
        )


def merge_mo_optimizer_results(results: Sequence[MultiObjectiveOptimizerResult]) -> MultiObjectiveOptimizerResult:
    """Union of the results, avoiding copies of results with the same hyperparams.
    Input objects are not modified."""
    res_hyperparams = {}
    for r in results:
        for p, h in zip(r.predictors(), r.hyperparams()):
            if h not in res_hyperparams:
                res_hyperparams[h] = p
    names = set([r.name() for r in results])
    nicks = set([r.nick() for r in results])
    hyperparams = list(res_hyperparams.keys())
    return MultiObjectiveOptimizerResult(
        name=sequence_to_string(li=names, brackets=False),
        nick=sequence_to_string(li=nicks, compact=True, brackets=False),
        predictors=[res_hyperparams[h] for h in hyperparams],
        hyperparams=hyperparams)
