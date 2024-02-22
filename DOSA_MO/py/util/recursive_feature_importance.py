from abc import ABC, abstractmethod
from collections.abc import Sequence

from pandas import DataFrame

from input_data.outcome import Outcome, OutcomeType
from model.mask import Mask
from model.masked_model import MaskedClassModel
from model.model_with_coef import ClassModelWithCoef
from model.logistic import LassoWithFallback
from util.distribution.average_distribution import AverageDistribution
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.utils import IllegalStateError


class MultiOutcomeRecursiveFeatureImportance(ABC):

    def next_fi(self, x: DataFrame, outcomes: Sequence[Outcome], feature_importance: Distribution,
                n_features_to_select: int, n_proc: int = 1
                ) -> Distribution:
        """Input and returned distributions are on the collapsed views x."""
        n_features = len(x.columns)
        if len(feature_importance) != n_features:
            raise ValueError()
        return self._inner_next_fi(x=x, outcomes=outcomes, feature_importance=feature_importance,
                                   n_features_to_select=n_features_to_select, n_proc=n_proc)

    @abstractmethod
    def _inner_next_fi(self, x: DataFrame, outcomes: Sequence[Outcome],
                       feature_importance: Distribution, n_features_to_select: int, n_proc=1
                       ) -> Distribution:
        raise NotImplementedError()


class RecursiveFeatureImportance(ABC):

    @abstractmethod
    def next_fi(self, x: DataFrame, y: DataFrame, feature_importance: Distribution,
                n_features_to_select: int, n_proc: int = 1
                ) -> Distribution:
        raise NotImplementedError()

    @abstractmethod
    def is_none(self) -> bool:
        raise NotImplementedError()


class RecursiveFeatureImportanceNone(RecursiveFeatureImportance):

    def next_fi(self, x: DataFrame, y: DataFrame, feature_importance: Distribution,
                n_features_to_select: int, n_proc: int = 1
                ) -> Distribution:
        raise IllegalStateError()

    def is_none(self) -> bool:
        return True


class RecursiveFeatureImportanceCoef(RecursiveFeatureImportance):
    __model: ClassModelWithCoef

    def __init__(self, model: ClassModelWithCoef):
        self.__model = model

    def next_fi(self, x: DataFrame, y: DataFrame, feature_importance: Distribution,
                n_features_to_select: int, n_proc: int = 1
                ) -> Distribution:
        n_cols = len(x.columns)
        len_dist = len(feature_importance)
        if n_cols != len_dist:
            raise ValueError(
                "Number of columns and length of distribution differ.\n" +
                "Number of columns: " + str(n_cols) + "\n" +
                "distribution length: " + str(len_dist) + "\n")
        mask = Mask.from_distribution(d=feature_importance)
        masked_model = MaskedClassModel(inner=self.__model, mask=mask)
        masked_predictor = masked_model.fit(x=x, y=y)
        fi = masked_predictor.feature_importance()
        return ConcreteDistribution(probs=fi).focus(n_features_to_select)

    def is_none(self) -> bool:
        return False


class RecursiveFeatureImportanceLasso(RecursiveFeatureImportanceCoef):

    def __init__(self):
        RecursiveFeatureImportanceCoef.__init__(self, model=LassoWithFallback())


class CompositeMultiOutcomeRecursiveFeatureImportance(MultiOutcomeRecursiveFeatureImportance):
    __fi_class: RecursiveFeatureImportance
    __fi_survival: RecursiveFeatureImportance

    def __init__(self,
                 class_fi: RecursiveFeatureImportance = RecursiveFeatureImportanceNone(),
                 survival_fi: RecursiveFeatureImportance = RecursiveFeatureImportanceNone()):
        self.__fi_class = class_fi
        self.__fi_survival = survival_fi

    def _inner_next_fi(self, x: DataFrame, outcomes: Sequence[Outcome], feature_importance: Distribution,
                       n_features_to_select: int, n_proc: int = 1) -> Distribution:
        distributions = []
        for o in outcomes:
            o_data = o.data()
            o_type = o.type()
            if o_type is OutcomeType.categorical:
                fi_to_use = self.__fi_class
            elif o_type is OutcomeType.survival:
                fi_to_use = self.__fi_survival
            else:
                raise ValueError("Unexpected outcome type.")
            if not fi_to_use.is_none():
                distributions.append(fi_to_use.next_fi(x=x, y=o_data, feature_importance=feature_importance,
                                                       n_features_to_select=n_features_to_select,
                                                       n_proc=n_proc))
        if len(distributions) == 0:
            res_distribution = feature_importance
        else:
            res_distribution = AverageDistribution(distributions)
        return res_distribution.focus(n_features_to_select)


class MultiOutcomeRecursiveFeatureImportanceLasso(CompositeMultiOutcomeRecursiveFeatureImportance):

    def __init__(self):
        CompositeMultiOutcomeRecursiveFeatureImportance.__init__(self,
                                                                 class_fi=RecursiveFeatureImportanceLasso())
