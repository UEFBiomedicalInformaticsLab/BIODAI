from input_data.class_and_surv_best_effort_input_creator import ClassAndSurvBestEffortInputCreator
from input_data.class_and_surv_input_creator import ClassAndSurvInputCreator
from input_data.class_input_creator import ClassInputCreator
from input_data.no_outcome_input_creator import NoOutcomeInputCreator

GLEASON_COL = "Reviewed_Gleason_category"
GLEASON_STR = "Gleason"

TCGA_BRCA_NORMAL_NICK = "brca_normal"
TCGA_DIG5_NICK = "tcga_dig5"
TCGA_DIG_TYPE_NICK = "tcga_dig_type"
EXT_DIG_NICK = "ext_dig"
TCGA_KIR3_NICK = "tcga_kir3"
TCGA_KIDNEY_NORMAL_NICK = "tcga_kidney_normal"
TCGA_KIDNEY_IHC_NICK = "kidney_ihc"
TCGA_KIDNEY_IHC_DET_NICK = "kidney_ihc_det"
TCGA_KIDNEY_IHC_DET_OS_NICK = "kidney_ihc_det_os"
CUSTOM_NICK = "custom"


INPUT_CREATORS_LIST = [
    ClassInputCreator(nick="tcga_dig", outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick=TCGA_DIG5_NICK, outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick=TCGA_DIG_TYPE_NICK, outcome_col="cancer.type", outcome_name="type"),
    ClassInputCreator(nick=EXT_DIG_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=TCGA_KIR3_NICK, outcome_col="PanKidney Pathology", outcome_name="type"),
    ClassInputCreator(nick=TCGA_KIDNEY_IHC_NICK, outcome_col="PanKidney Pathology", outcome_name="type"),
    ClassInputCreator(nick=TCGA_KIDNEY_IHC_DET_NICK, outcome_col="PanKidney Pathology", outcome_name="type"),
    NoOutcomeInputCreator(nick=TCGA_BRCA_NORMAL_NICK),
    NoOutcomeInputCreator(nick=TCGA_KIDNEY_NORMAL_NICK),
    ClassAndSurvInputCreator(nick=TCGA_KIDNEY_IHC_DET_OS_NICK, class_outcome_col="PanKidney Pathology"),
    ClassAndSurvBestEffortInputCreator(nick=CUSTOM_NICK, class_outcome_col="type")
]

INPUT_CREATORS_DICT = {}
for i in INPUT_CREATORS_LIST:
    INPUT_CREATORS_DICT[i.nick()] = i
