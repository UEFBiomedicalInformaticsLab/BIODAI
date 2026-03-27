from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy


DEFAULT_LABELS_MAP = {}  # Order of mappings is important.
DEFAULT_LABELS_MAP["NB_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["RF_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["svm_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["logit100_bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["bal_acc"] = "balanced accuracy"
DEFAULT_LABELS_MAP["root_leanness"] = "number of features"
DEFAULT_LABELS_MAP["leanness"] = "number of features"
DEFAULT_LABELS_MAP["min_separation"] = "min separation"
DEFAULT_LABELS_MAP["root_separation"] = "root separation"
DEFAULT_LABELS_MAP["SKSurvCox_c-index"] = "concordance index"  # Was "SKSurv Cox c-index"
DEFAULT_LABELS_MAP["inner_cv_NB_brier"] = "estimated 1 - Brier score"
DEFAULT_LABELS_MAP["NB_brier"] = "1 - Brier score"
DEFAULT_LABELS_MAP["tcga_brca"] = "TCGA-breast"
DEFAULT_LABELS_MAP["tcga_ki_ihc_os"] = "TCGA-kidney"
DEFAULT_LABELS_MAP["tcga_ki_ihc_det"] = "TCGA-kidney"
DEFAULT_LABELS_MAP["swedish"] = "SCAN-B"


class LabelsTransformer:
    __labels_map: dict[str, str]
    __capitalize_first_letter: bool

    def __init__(self, labels_map: dict[str, str], capitalize_first_letter: bool = False):
        self.__labels_map = labels_map
        self.__capitalize_first_letter = capitalize_first_letter

    def apply(self, label: str) -> str:
        for a in self.__labels_map:
            label = label.replace(a, self.__labels_map[a])
        if self.__capitalize_first_letter:
            res = ""
            if len(label) > 0:
                res += label[0].capitalize()
            if len(label) > 1:
                res += label[1:]
            return res
        else:
            return label

    def add(self, from_str: str, to_str: str) -> LabelsTransformer:
        new_map = copy(self.__labels_map)
        new_map[from_str] = to_str
        return LabelsTransformer(labels_map=new_map, capitalize_first_letter=self.__capitalize_first_letter)

    def apply_all(self, strings: Iterable[str]) -> Sequence[str]:
        return [self.apply(s) for s in strings]


DUMMY_LABELS_TRANSFORMER = LabelsTransformer(labels_map={}, capitalize_first_letter=False)
DEFAULT_LABELS_TRANSFORMER = LabelsTransformer(labels_map=DEFAULT_LABELS_MAP, capitalize_first_letter=True)
NO_LOG_LABELS_TRANSFORMER = DEFAULT_LABELS_TRANSFORMER.add(from_str="log_", to_str="")
