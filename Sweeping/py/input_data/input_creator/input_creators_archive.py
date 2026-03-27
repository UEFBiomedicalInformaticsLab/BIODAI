from input_data.input_creator.class_and_surv_best_effort_input_creator import ClassAndSurvBestEffortInputCreator


TCGA_PRAD_NICK = "prad"
EXT_DIG_NICK = "ext_dig"
CPTAC3_SUB_NICK = "cptac3_sub"
CPTAC3_SUB_UQ_NICK = "cptac3_sub_uq"
CPTAC3_SUB_UQ2_NICK = "cptac3_sub_uq2"
CPTAC3_SUB_UQ3_NICK = "cptac3_sub_uq3"
CPTAC3_SUB_UQ4_NICK = "cptac3_sub_uq4"
GSE138042_NO0_NICK = "gse138042_no0"
EXT_OV_NICK = "ext_ov"
EXT_OV2_NICK = "ext_ov2"
KID_GSE152938_NICK = "kid_gse152938"
KID_GSE152938D_NICK = "kid_gse152938d"
KID_GSE152938D_T2_NICK = "kid_gse152938dT2"
KID_GSE152938B1_NICK = "kid_gse152938b1"
KID_GSE152938B2_NICK = "kid_gse152938b2"
SWEDISH_NEW_NICK = "swedish_new"
SWEDISH_NEW2_NICK = "swedish_new2"
CUSTOM_NICK = "custom"
KIRC_MV_NICK = "kirc_mv"
SARC_MV_NICK = "sarc_mv"
LGG_MV_NICK = "lgg_mv"
ORIGINAL_KIRC_MV_NICK = "original_kirc_mv"


INPUT_CREATORS_LIST = [
    ClassAndSurvBestEffortInputCreator(nick=CUSTOM_NICK, class_outcome_col="type"),
    ClassAndSurvBestEffortInputCreator(nick=KIRC_MV_NICK, class_outcome_col="type", name="KIRC"),
    ClassAndSurvBestEffortInputCreator(nick=SARC_MV_NICK, class_outcome_col="type", name="SARC"),
    ClassAndSurvBestEffortInputCreator(nick=LGG_MV_NICK, class_outcome_col="type", name="LGG")
]

INPUT_CREATORS_DICT = {}
for i in INPUT_CREATORS_LIST:
    INPUT_CREATORS_DICT[i.nick()] = i
