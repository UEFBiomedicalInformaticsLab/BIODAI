from input_data.input_creator.input_creators_archive import TCGA_PRAD_NICK
from load_omics_views import MIRNA_NAME, MRNA_NAME, RPPAA_NAME, METH_NAME, AGE_NAME, LOG_MRNA_NAME, CLINIC_NAME, \
    LOG_MIRNA_NAME, SNP_NAME, PROTEOMICS_NAME
from objective.objective_with_importance.objective_computer_with_importance import BALANCED_ACCURACY_NICK
from objective.objective_with_importance.leanness import LEANNESS_NICK
from views.adjusted_view_definition import AdjustedViewDef

RESAMPLED_SWEEPING_NAME = "sweeping_ga"
FAT_CONCATENATED_SWEEPING_NAME = "csweeping_ga"
LEAN_CONCATENATED_SWEEPING_NAME = "lcsweeping_ga"
NSGA_STAR_NAME = "classic_ga"
LASSO_NAME = "lasso"
RIDGE_NAME = "ridge"
LASSO_MO_NAME = "lasso_mo"
GUIDED_FORWARD_NAME = "guided_forward"
RFE_NAME = "rfe"
PAM50_NAME = "PAM50"
ADJUSTED_NAME = "adjusted"
DEFAULT_ALGORITHM_NAME = RESAMPLED_SWEEPING_NAME

NAIVE_BAYES_NAME = "naive_bayes"
LOGISTIC_NAME = "logistic"
DEFAULT_MODEL_NAME = NAIVE_BAYES_NAME

SOCIAL_SPACE_NAME = "social_space"
CROWDING_DISTANCE_NAME = "crowding_distance"
SOCIAL_SPACE_FULL_NAME = "social_space_full"
CROWDING_DISTANCE_FULL_NAME = "crowding_distance_full"
SOCIAL_SPACE_CI_NAME = "social_space_clone_index"
CROWDING_DISTANCE_CI_NAME = "crowding_distance_clone_index"
NSGA3_NAME = "nsga3"
NSGA3_CI_NAME = "nsga3_clone_index"
SORTING_STRATEGY_DEFAULT = SOCIAL_SPACE_NAME

NONE_NAME = "none"
UNIFORM_NAME = "uniform"
SOFT_LASSO_NAME = "soft_lasso"
ANOVA_NAME = "anova"
COX_NAME = "cox"
UNIVARIATE_COX_NAME = "univariate_cox"
UNIVARIATE_NB_NAME = "univariate_nb"
CNVSNP = "cnvsnp"

DEFAULT_OBJECTIVE_NAMES = [LEANNESS_NICK, BALANCED_ACCURACY_NICK]

DEFAULT_DATASET_NAME = TCGA_PRAD_NICK
VIEW_NAMES = [MIRNA_NAME, MRNA_NAME, LOG_MIRNA_NAME, LOG_MRNA_NAME, RPPAA_NAME, CNVSNP, METH_NAME, AGE_NAME,
              CLINIC_NAME, SNP_NAME, PROTEOMICS_NAME]
DEFAULT_VIEWS_MV = AdjustedViewDef.create_unadjusted(view_names=VIEW_NAMES)

INITIAL_FEATURES_UNIFORM_NAME = "uniform"
INITIAL_FEATURES_BINOMIAL_NAME = "binomial"
INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME = "binomial_from_uniform"
DEFAULT_INITIAL_FEATURES_STRATEGY_NAME = INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME

STRATIFIED_K_FOLD_NAME = "stratified_k_fold"
LOAD_FOLD_NAME = "load"
AUTO_FOLD_NAME = "auto"
DEFAULT_OUTER_FOLDS_NAME = AUTO_FOLD_NAME
