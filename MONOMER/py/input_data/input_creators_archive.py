from input_data.brca_input_creator import OUTCOME_PAM50_NAME
from input_data.class_input_creator import ClassInputCreator

GLEASON_COL = "Reviewed_Gleason_category"
GLEASON_STR = "Gleason"

TCGA_BRCA_MRNA_METH = "brca_mrna_meth"
TCGA_BRCA_MRNA_MIRNA = "brca_mrna_mirna"
TCGA_LUNG_MRNA_MIRNA = "lung_mrna_mirna"
TCGA_OV_NICK = "tcga_ov"
TCGA_THCA2_NICK = "tcga_thca2"
TCGA_THCA3_NICK = "tcga_thca3"
TCGA_THCA_BL_NICK = "tcga_thca_bl"
TCGA_DIG5_NICK = "tcga_dig5"
TCGA_DIG_TYPE_NICK = "tcga_dig_type"
TCGA_LU2_NICK = "tcga_lu2"
EXT_DIG_NICK = "ext_dig"
CPTAC3_NICK = "cptac3"
CPTAC3_SUB_NICK = "cptac3_sub"
CPTAC3_UQ_NICK = "cptac3_uq"
CPTAC3_SUB_UQ_NICK = "cptac3_sub_uq"
CPTAC3_SUB_UQ2_NICK = "cptac3_sub_uq2"
CPTAC3_SUB_UQ3_NICK = "cptac3_sub_uq3"
CPTAC3_SUB_UQ4_NICK = "cptac3_sub_uq4"
GSE138042_NICK = "gse138042"
GSE138042_NO0_NICK = "gse138042_no0"
GSE138042_NO0B_NICK = "gse138042_no0b"
EXT_OV_NICK = "ext_ov"
EXT_OV2_NICK = "ext_ov2"
KID_GSE152938_NICK = "kid_gse152938"
KID_GSE152938B_NICK = "kid_gse152938b"
KID_GSE152938C_NICK = "kid_gse152938c"
KID_GSE152938D_NICK = "kid_gse152938d"
KID_GSE152938B1_NICK = "kid_gse152938b1"
KID_GSE152938B2_NICK = "kid_gse152938b2"
TCGA_KIR3_NICK = "tcga_kir3"
SWEDISH_NEW_NICK = "swedish_new"
SWEDISH_NEW2_NICK = "swedish_new2"


INPUT_CREATORS_LIST = [
    ClassInputCreator(nick="prad_mrna", outcome_col=GLEASON_COL, outcome_name=GLEASON_STR),
    ClassInputCreator(nick="prad_mrna3", outcome_col=GLEASON_COL, outcome_name=GLEASON_STR),
    ClassInputCreator(nick="prad_mrna3b", outcome_col=GLEASON_COL, outcome_name=GLEASON_STR),
    ClassInputCreator(nick="prad_mrna2", outcome_col=GLEASON_COL, outcome_name=GLEASON_STR),
    ClassInputCreator(nick="prad_mrna2full", outcome_col=GLEASON_COL, outcome_name=GLEASON_STR),
    ClassInputCreator(nick=TCGA_BRCA_MRNA_METH, outcome_col="Pam50", outcome_name="Pam50"),
    ClassInputCreator(nick=TCGA_BRCA_MRNA_MIRNA, outcome_col="Pam50", outcome_name="Pam50"),
    ClassInputCreator(nick=TCGA_LUNG_MRNA_MIRNA, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=TCGA_OV_NICK, outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick="tcga_thca", outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=TCGA_THCA3_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=TCGA_THCA_BL_NICK, outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick="tcga_dig", outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick=TCGA_DIG5_NICK, outcome_col="Subtype_Selected", outcome_name="type"),
    ClassInputCreator(nick=TCGA_DIG_TYPE_NICK, outcome_col="cancer.type", outcome_name="type"),
    ClassInputCreator(nick=TCGA_LU2_NICK, outcome_col="type", outcome_name="type"),
    ClassInputCreator(nick=EXT_DIG_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=CPTAC3_NICK, outcome_col="type", outcome_name="type"),
    ClassInputCreator(nick=CPTAC3_SUB_NICK, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=CPTAC3_UQ_NICK, outcome_col="type", outcome_name="type"),
    ClassInputCreator(nick=CPTAC3_SUB_UQ_NICK, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=CPTAC3_SUB_UQ2_NICK, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=CPTAC3_SUB_UQ3_NICK, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=CPTAC3_SUB_UQ4_NICK, outcome_col="expression_subtype", outcome_name="expression_subtype"),
    ClassInputCreator(nick=GSE138042_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=TCGA_THCA2_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=EXT_OV_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=EXT_OV2_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=GSE138042_NO0_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=GSE138042_NO0B_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938B_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938B1_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938B2_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938C_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=KID_GSE152938D_NICK, outcome_col="outcome", outcome_name="type"),
    ClassInputCreator(nick=TCGA_KIR3_NICK, outcome_col="PanKidney Pathology", outcome_name="type"),
    ClassInputCreator(nick=SWEDISH_NEW_NICK, outcome_col="Pam50", outcome_name=OUTCOME_PAM50_NAME),
    ClassInputCreator(nick=SWEDISH_NEW2_NICK, outcome_col="Pam50", outcome_name=OUTCOME_PAM50_NAME)
]

INPUT_CREATORS_DICT = {}
for i in INPUT_CREATORS_LIST:
    INPUT_CREATORS_DICT[i.nick()] = i
