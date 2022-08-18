from model.multi_view_model import MVPredictor
from util.preconditions import check_none
from util.sequence_utils import str_in_lines


class WorkerResult:
    __fit: list
    __predictors: [MVPredictor]

    def __init__(self, fit: list, predictors: [MVPredictor]):
        """A predictor can be none if the related fitness does not need a predictor."""
        self.__fit = check_none(fit)
        self.__predictors = check_none(predictors)
        for f in fit:
            if not isinstance(f, float):
                raise ValueError("Passed values: " + str(fit))

    def fit(self) -> list:
        return self.__fit

    def predictors(self) -> [MVPredictor]:
        return self.__predictors

    def __str__(self):
        return "Fitness: " + str(self.__fit) + "\n" +\
               "Predictors:\n" + str_in_lines(self.__predictors)
