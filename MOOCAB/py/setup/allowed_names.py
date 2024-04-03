from input_data.input_creators_archive import TCGA_KIDNEY_IHC_DET_OS_NICK
from load_omics_views import MRNA_NAME
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import SoftLeanness

NSGA_STAR_NAME = "classic_ga"
LASSO_NAME = "lasso"
RIDGE_NAME = "ridge"
DEFAULT_ALGORITHM_NAME = NSGA_STAR_NAME

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
CNVSNP = "cnvsnp"

DEFAULT_OBJECTIVE_NAMES = [SoftLeanness().nick(), BalancedAccuracy().nick()]

DEFAULT_DATASET_NAME = TCGA_KIDNEY_IHC_DET_OS_NICK
VIEW_NAMES = [MRNA_NAME]
DEFAULT_VIEWS = [MRNA_NAME]
DEFAULT_VIEWS_MV = VIEW_NAMES

INITIAL_FEATURES_UNIFORM_NAME = "uniform"
INITIAL_FEATURES_BINOMIAL_NAME = "binomial"
INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME = "binomial_from_uniform"
DEFAULT_INITIAL_FEATURES_STRATEGY_NAME = INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME

STRATIFIED_K_FOLD_NAME = "stratified_k_fold"
LOAD_FOLD_NAME = "load"
AUTO_FOLD_NAME = "auto"
DEFAULT_OUTER_FOLDS_NAME = AUTO_FOLD_NAME
