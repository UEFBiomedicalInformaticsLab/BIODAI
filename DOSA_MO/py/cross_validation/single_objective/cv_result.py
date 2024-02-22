from typing import Optional, Sequence

from model.mv_predictor import MVPredictor
from util.hyperbox.hyperbox import Interval
from util.sequence_utils import sequence_to_string
from util.utils import IllegalStateError


class CVResult:
    __fitness: float
    __importances: Optional[Sequence[float]]
    __std_dev: Optional[float]
    __ci95: Optional[Interval]
    __bootstrap_mean: Optional[float]
    __final_predictor: Optional[MVPredictor]

    def __init__(self,
                 fitness: float,
                 importances: Optional[Sequence[float]] = None,
                 std_dev: Optional[float] = None,
                 ci95: Optional[Interval] = None,
                 bootstrap_mean: Optional[float] = None,
                 final_predictor: Optional[MVPredictor] = None):
        self.__fitness = fitness
        self.__importances = importances
        self.__std_dev = std_dev
        self.__ci95 = ci95
        self.__bootstrap_mean = bootstrap_mean
        self.__final_predictor = final_predictor

    def fitness(self) -> float:
        return self.__fitness

    def has_importances(self) -> bool:
        return self.__importances is not None

    def importances(self) -> Sequence[float]:
        if self.has_importances():
            return self.__importances
        else:
            raise IllegalStateError()

    def set_importances(self, importances: Sequence[float]):
        self.__importances = importances

    def reset_importances(self):
        self.__importances = None

    def has_std_dev(self) -> bool:
        return self.__std_dev is not None

    def std_dev(self) -> float:
        if self.has_std_dev():
            return self.__std_dev
        else:
            raise IllegalStateError("Std dev not present.\n" + "Self:\n" + str(self))

    def set_std_dev(self, std_dev: float):
        self.__std_dev = std_dev

    def has_ci95(self) -> bool:
        return self.__ci95 is not None

    def ci95(self) -> Interval:
        if self.has_ci95():
            return self.__ci95
        else:
            raise IllegalStateError()

    def set_ci95(self, ci95: Interval):
        self.__ci95 = ci95

    def has_bootstrap_mean(self) -> bool:
        return self.__bootstrap_mean is not None

    def bootstrap_mean(self) -> float:
        if self.has_bootstrap_mean():
            return self.__bootstrap_mean
        else:
            raise IllegalStateError("Bootstrap mean not present.\n" + "Self:\n" + str(self))

    def set_bootstrap_mean(self, bootstrap_mean: float):
        self.__bootstrap_mean = bootstrap_mean

    def has_final_predictor(self) -> bool:
        return self.__final_predictor is not None

    def final_predictor(self) -> MVPredictor:
        if self.has_final_predictor():
            return self.__final_predictor
        else:
            raise IllegalStateError()

    def set_final_predictor(self, predictor: MVPredictor):
        self.__final_predictor = predictor

    def __str__(self) -> str:
        res = "Fitness: " + str(self.__fitness) + "\n"
        if self.has_std_dev():
            res += "Standard deviation: " + str(self.std_dev()) + "\n"
        if self.has_ci95():
            res += "Confidence interval 95%: " + str(self.ci95()) + "\n"
        if self.has_bootstrap_mean():
            res += "Bootstrap mean: " + str(self.bootstrap_mean()) + "\n"
        if self.has_final_predictor():
            res += "Final predictor: " + str(self.final_predictor()) + "\n"
        if self.has_importances():
            res += "Feature importances: " + sequence_to_string(self.importances()) + "\n"
        return res
