from feature_importance.feature_importance_uniform import FeatureImportanceUniform
from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance, MVFeatureImportanceUniform
from input_data.input_data import InputData
from input_data.outcome_type import OutcomeType
from util.dict_utils import merge_common_keys
from util.distribution.average_distribution import AverageDistribution
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, NULL_PRINTER
from util.randoms import random_state_context_blended


class MultiOutcomeFeatureImportance(MultiViewFeatureImportance):
    __mv_fi_class: MultiViewFeatureImportance
    __mv_fi_survival: MultiViewFeatureImportance

    def __init__(self, class_fi: MultiViewFeatureImportance = MVFeatureImportanceUniform(),
                 survival_fi: MultiViewFeatureImportance = MVFeatureImportanceUniform()):
        self.__mv_fi_class = class_fi
        self.__mv_fi_survival = survival_fi

    def compute(
            self, input_data: InputData, n_proc: int = 1, printer: Printer = NULL_PRINTER) -> dict[str,Distribution]:
        """Uses a separate random context and restores the current one at the end."""
        if input_data.needs_adjustment():
            raise ValueError("Input data needs to be adjusted before computing the feature importance.")
        with random_state_context_blended(additional_seed=432, printer=printer):
            distributions_fi_view: list[dict[str,Distribution]] = []
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
                    distributions_fi_view.append(
                        fi_to_use.compute(input_data=single_outcome_input, n_proc=n_proc, printer=printer))
            if len(distributions_fi_view) == 0:
                views = input_data.views_dict()
                fi_uniform = FeatureImportanceUniform()
                return {v: fi_uniform.compute(x=views[v]) for v in views}
            else:
                distributions_view_fi = merge_common_keys(dicts=distributions_fi_view)
                return {view: AverageDistribution(distributions)
                        for view, distributions in distributions_view_fi.items()}

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
