from collections.abc import Sequence
from typing import Optional

import deprecation

from model.mv_predictor import MVPredictor
from util.preconditions import check_none
from util.sequence_utils import str_in_lines
from util.utils import IllegalStateError


@deprecation.deprecated(details="Not used anymore in favor of MOCVResult.")
class WorkerResult:
    """Not used anymore in favor of MOCVResult."""
    __fit: list[float]
    __predictors: [MVPredictor]
    __importances: Optional[Sequence[float]]

    def __init__(self, fit: list[float], predictors: [MVPredictor], importance: Optional[Sequence[float]]):
        """A predictor can be none if the related fitness does not need a predictor.
        Importance is one value for each active feature."""
        self.__fit = check_none(fit)
        self.__predictors = check_none(predictors)
        self.__importances = importance
        for f in fit:
            if not isinstance(f, float):
                raise ValueError("Passed values: " + str(fit))

    def fit(self) -> list:
        return self.__fit

    def predictors(self) -> [MVPredictor]:
        return self.__predictors

    def has_importances(self) -> bool:
        return self.__importances is not None

    def importances(self) -> Sequence[float]:
        """The elements of the distribution are the active features of the individual, in the same order."""
        if self.has_importances():
            return self.__importances
        else:
            raise IllegalStateError()

    def __str__(self):
        return "Fitness: " + str(self.__fit) + "\n" +\
               "Predictors:\n" + str_in_lines(self.__predictors) + "\n" +\
               "Feature importance:\n" + str(self.__importances) + "\n"
