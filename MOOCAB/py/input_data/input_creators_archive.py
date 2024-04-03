from input_data.class_and_surv_best_effort_input_creator import ClassAndSurvBestEffortInputCreator
from input_data.class_and_surv_input_creator import ClassAndSurvInputCreator
from input_data.class_input_creator import ClassInputCreator


TCGA_KIDNEY_IHC_DET_NICK = "kidney_ihc_det"
TCGA_KIDNEY_IHC_DET_OS_NICK = "kidney_ihc_det_os"
CUSTOM_NICK = "custom"


INPUT_CREATORS_LIST = [
    ClassInputCreator(nick=TCGA_KIDNEY_IHC_DET_NICK, outcome_col="PanKidney Pathology", outcome_name="type"),
    ClassAndSurvInputCreator(nick=TCGA_KIDNEY_IHC_DET_OS_NICK, class_outcome_col="PanKidney Pathology"),
    ClassAndSurvBestEffortInputCreator(nick=CUSTOM_NICK, class_outcome_col="type")
]

INPUT_CREATORS_DICT = {}
for i in INPUT_CREATORS_LIST:
    INPUT_CREATORS_DICT[i.nick()] = i
