DEFAULT_LABELS_MAP = {}
DEFAULT_LABELS_MAP["bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["NB_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["RF_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["logit100_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["leanness"] = "number of features"
DEFAULT_LABELS_MAP["root_leanness"] = "number of features"
DEFAULT_LABELS_MAP["min_separation"] = "min separation"
DEFAULT_LABELS_MAP["root_separation"] = "root separation"
DEFAULT_LABELS_MAP["SKSurvCox_c-index"] = "SKSurv Cox c-index"


class LabelsTransformer:
    __labels_map: dict[str, str]

    def __init__(self, labels_map: dict[str, str], capitalize_first_letter: bool = False):
        self.__labels_map = labels_map
        self.__capitalize_first_letter = capitalize_first_letter

    def apply(self, label: str) -> str:
        if label in self.__labels_map:
            res = self.__labels_map[label]
        else:
            res = label
        if self.__capitalize_first_letter:
            res = res[0].capitalize() + res[1:]
        return res


DUMMY_LABELS_TRANSFORMER = LabelsTransformer(labels_map={}, capitalize_first_letter=False)
DEFAULT_LABELS_TRANSFORMER = LabelsTransformer(labels_map=DEFAULT_LABELS_MAP, capitalize_first_letter=True)
