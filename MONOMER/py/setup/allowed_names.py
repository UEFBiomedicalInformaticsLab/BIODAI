from input_data.prad_input_creator import PradInputCreator
from load_omics_views import MIRNA_NAME, MRNA_NAME, RPPAA_NAME, METH_NAME
from model.forest import ForestWithFallback
from objective.objective_computer import Leanness, BalancedAccuracy

SWEEPING_NAME = "sweeping_ga"
NSGA_STAR_NAME = "classic_ga"
LASSO_NAME = "lasso"
LASSO_MO_NAME = "lasso_mo"
GUIDED_FORWARD_NAME = "guided_forward"
RFE_NAME = "rfe"
DEFAULT_ALGORITHM_NAME = SWEEPING_NAME

NAIVE_BAYES_NAME = "naive_bayes"
LOGISTIC_NAME = "logistic"
FOREST_NAME = ForestWithFallback().nick()
DEFAULT_MODEL_NAME = NAIVE_BAYES_NAME

SOCIAL_SPACE_NAME = "social_space"
CROWDING_DISTANCE_NAME = "crowding_distance"
SOCIAL_SPACE_FULL_NAME = "social_space_full"
CROWDING_DISTANCE_FULL_NAME = "crowding_distance_full"
SOCIAL_SPACE_CI_NAME = "social_space_clone_index"
CROWDING_DISTANCE_CI_NAME = "crowding_distance_clone_index"
NSGA3_NAME = "nsga3"
SORTING_STRATEGY_DEFAULT = SOCIAL_SPACE_NAME

NONE_NAME = "none"
UNIFORM_NAME = "uniform"
SOFT_LASSO_NAME = "soft_lasso"
ANOVA_NAME = "anova"
COX_NAME = "cox"
UNIVARIATE_COX_NAME = "univariate_cox"

DEFAULT_OBJECTIVE_NAMES = [Leanness().nick(), BalancedAccuracy().nick()]

DEFAULT_DATASET_NAME = PradInputCreator().nick()
VIEW_NAMES = [MIRNA_NAME, MRNA_NAME, RPPAA_NAME, METH_NAME]
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
