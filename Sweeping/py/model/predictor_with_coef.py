from abc import abstractmethod, ABC
from typing import Union, Sequence

from feature_importance.feature_importance_by_lasso import collapse_coef
from model.importance_extractor import SklearnImportanceExtractor, OffImportanceExtractor
from model.coef_extractor import SklearnCoefExtractor
from model.sv_model import SVPredictor, SklearnSVPredictorWrapper, PredictStrategy, JustPredictStrategy
from util.math.list_math import list_abs
from util.utils import IllegalStateError
from util.str_utils import name_value


class SVPredictorWithCoef(SVPredictor):

    @abstractmethod
    def coef(self) -> Union[Sequence[Sequence[float]], Sequence[float]]:
        """Classifiers have a sequence for each class, regressors just one sequence."""
        raise NotImplementedError()

    def feature_importance(self) -> Sequence[float]:
        return collapse_coef(self.coef())

    def __str__(self) -> str:
        res = "predictor with coefficients\n"
        res += self.coefs_str() + "\n"
        return res

    def coefs_str(self) -> str:
        coefs = self.coef()
        try:
            if len(coefs) > 0:
                if len(coefs) <= 5:
                    if len(coefs[0]) < 5:
                        return name_value(name="coefficients", value=coefs)
                    else:
                        return "Many coefficients"
                return "Many coefficients"
            else:
                return "Zero coefficients"
        except BaseException:
            raise IllegalStateError("Coefficients do not work as a sequence of sequences: " + str(coefs) + "\n")


class SVPredictorWithClassCoef(SVPredictorWithCoef, ABC):

    @abstractmethod
    def coef(self) -> Sequence[Sequence[float]]:
        """Classifiers have a sequence for each class."""
        raise NotImplementedError()

    def __str__(self) -> str:
        res = "predictor with coefficients\n"
        res += self.coefs_str() + "\n"
        return res




class SklearnPredictorWrapperWithCoef(SklearnSVPredictorWrapper, SVPredictorWithCoef, ABC):
    pass


class SklearnPredictorWrapperWithExtractor(SklearnSVPredictorWrapper, SVPredictorWithCoef):
    __coef_extractor: SklearnCoefExtractor
    __importance_extractor: SklearnImportanceExtractor

    def __init__(self, sklearn_predictor, coef_extractor: SklearnCoefExtractor,
                 importance_extractor: SklearnImportanceExtractor = OffImportanceExtractor(),
                 predict_strategy: PredictStrategy = JustPredictStrategy()):
        SklearnSVPredictorWrapper.__init__(self, sklearn_predictor=sklearn_predictor, predict_strategy=predict_strategy)
        self.__coef_extractor = coef_extractor
        self.__importance_extractor = importance_extractor

    def coef(self) -> Sequence[Sequence[float]]:
        """Raises exception if cannot extract."""
        return self.__coef_extractor.extract_coef(self._get_sklearn_predictor())

    def feature_importance(self) -> Sequence[float]:
        if self.__importance_extractor.can_extract_importance():
            return self.__importance_extractor.extract_importance(self._get_sklearn_predictor())
        else:
            if self.__coef_extractor.can_extract_coef():
                coefs = self.__coef_extractor.extract_coef(self._get_sklearn_predictor())
                if len(coefs) == 0:
                    return []
                else:
                    if isinstance(coefs[0], float):
                        return list_abs(coefs)
                    else:
                        return collapse_coef(self.coef())
            else:
                raise IllegalStateError()

    def importance_str(self) -> str:
        if self.__importance_extractor.can_extract_importance():
            importance = self.__importance_extractor.extract_importance(self._get_sklearn_predictor())
            if len(importance) > 0:
                if len(importance) <= 5:
                    return name_value(name="feature importances", value=importance)
                else:
                    return "Many feature importances."
            else:
                return "Zero feature importances."
        else:
            return "Cannot extract feature importances."

    def __str__(self) -> str:
        res = ""
        res += SklearnSVPredictorWrapper.__str__(self) + "\n"
        if self.__coef_extractor.can_extract_coef():
            res += self.coefs_str() + "\n"
        elif self.__importance_extractor.can_extract_importance():
            res += self.importance_str() + "\n"
        return res
