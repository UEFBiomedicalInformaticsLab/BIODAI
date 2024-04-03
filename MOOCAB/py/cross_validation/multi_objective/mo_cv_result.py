from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

from cross_validation.single_objective.cv_result import CVResult
from model.mv_predictor import MVPredictor
from util.hyperbox.hyperbox import Interval
from util.math.list_math import list_add_all
from util.sequence_utils import sequence_to_string
from util.utils import IllegalStateError


class MOCVResult:
    """
    A predictor can be none if the related fitness does not need a predictor.
    Importance is one value for each active feature.
    Feature importances are not kept separately for each objective but instead are summed."""
    __so_results: Sequence[CVResult]
    __importances: Optional[Sequence[float]]

    def __init__(self, so_results: Sequence[CVResult]):
        self.__so_results = [deepcopy(so) for so in so_results]
        importances = []
        for cv_result in self.__so_results:
            if cv_result.has_importances():
                importances.append(cv_result.importances())
                cv_result.reset_importances()
        if len(importances) > 0:
            self.__importances = list_add_all(importances)
        else:
            self.__importances = None

    def n_objectives(self) -> int:
        return len(self.__so_results)

    def fit(self) -> list[float]:
        return [so.fitness()  for so in self.__so_results]

    def has_importances(self) -> bool:
        return self.__importances is not None

    def importances(self) -> Sequence[float]:
        if self.has_importances():
            return self.__importances
        else:
            raise IllegalStateError()

    def so_result(self, i: int) -> CVResult:
        return self.__so_results[i]

    def predictors(self) -> Sequence[MVPredictor]:
        predictors = []
        for cv_result in self.__so_results:
            if cv_result.has_final_predictor():
                predictors.append(cv_result.final_predictor())
            else:
                predictors.append(None)
        return predictors

    def __str__(self) -> str:
        res = "Multi-objective cross-validation result with " + str(self.n_objectives()) + " objectives.\n"
        for so in self.__so_results:
            res += str(so)
        if self.has_importances():
            res += "Feature importances: " + sequence_to_string(self.importances()) + "\n"
        return res

    def std_dev(self) -> Sequence[Optional[float]]:
        res = []
        for so in self.__so_results:
            if so.has_std_dev():
                res.append(so.std_dev())
            else:
                res.append(None)
        return res

    def ci95(self) -> Sequence[Optional[Interval]]:
        res = []
        for so in self.__so_results:
            if so.has_ci95():
                res.append(so.ci95())
            else:
                res.append(None)
        return res

    def bootstrap_mean(self):
        res = []
        for so in self.__so_results:
            if so.has_bootstrap_mean():
                res.append(so.bootstrap_mean())
            else:
                res.append(None)
        return res
