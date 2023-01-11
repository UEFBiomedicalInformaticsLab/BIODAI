from feature_importance.feature_importance_uniform import FeatureImportanceUniform
from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance, MVFeatureImportanceUniform
from input_data.input_data import InputData
from input_data.outcome import OutcomeType
from util.distribution.average_distribution import AverageDistribution
from util.distribution.distribution import Distribution
from util.sequence_utils import transpose


class MultiOutcomeFeatureImportance(MultiViewFeatureImportance):
    __mv_fi_class: MultiViewFeatureImportance
    __mv_fi_survival: MultiViewFeatureImportance

    def __init__(self, class_fi: MultiViewFeatureImportance = MVFeatureImportanceUniform(),
                 survival_fi: MultiViewFeatureImportance = MVFeatureImportanceUniform()):
        self.__mv_fi_class = class_fi
        self.__mv_fi_survival = survival_fi

    def compute(self, input_data: InputData, n_proc: int = 1) -> list[Distribution]:
        distributions_fi_view = []
        for o in input_data.outcomes():
            single_outcome_input = input_data.select_outcomes(keys=[o.name()])
            o_type = o.type()
            if o_type is OutcomeType.categorical:
                fi_to_use = self.__mv_fi_class
            elif o_type is OutcomeType.survival:
                fi_to_use = self.__mv_fi_survival
            else:
                raise ValueError("Unexpected outcome type.")
            if not fi_to_use.is_none():
                distributions_fi_view.append(fi_to_use.compute(single_outcome_input, n_proc=n_proc))
        if len(distributions_fi_view) == 0:
            views = input_data.views()
            fi_uniform = FeatureImportanceUniform()
            return [fi_uniform.compute(x=views[v]) for v in views]
        else:
            distributions_view_fi = transpose(x=distributions_fi_view)
            return [AverageDistribution(view) for view in distributions_view_fi]

    def is_none(self) -> bool:
        return False

    def name(self) -> str:
        return "(" + self.__mv_fi_class.name() + ", " + self.__mv_fi_survival.name() + ")"

    def nick(self) -> str:
        return "(" + self.__mv_fi_class.nick() + "," + self.__mv_fi_survival.nick() + ")"

    def __str__(self) -> str:
        res = "multi-objective feature importance\n"
        res += "Feature importance for classification: " + str(self.__mv_fi_class) + "\n"
        res += "Feature importance for survival: " + str(self.__mv_fi_survival) + "\n"
        return res
