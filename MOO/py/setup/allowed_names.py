from input_data.brca_input_creator import BrcaInputCreator
from load_omics_views import MIRNA_NAME, MRNA_NAME, RPPAA_NAME
from model.forest import ForestWithFallback
from objective.objective_computer import Leanness, BalancedAccuracy

NSGA_STAR_NAME = "classic_ga"
LASSO_NAME = "lasso"
LASSO_MO_NAME = "lasso_mo"
DEFAULT_ALGORITHM_NAME = NSGA_STAR_NAME

NAIVE_BAYES_NAME = "naive_bayes"
LOGISTIC_NAME = "logistic"
FOREST_NAME = ForestWithFallback().nick()
DEFAULT_MODEL_NAME = NAIVE_BAYES_NAME

CROWDING_DISTANCE_NAME = "crowding_distance"
CROWDING_DISTANCE_FULL_NAME = "crowding_distance_full"
CROWDING_DISTANCE_CI_NAME = "crowding_distance_clone_index"
SORTING_STRATEGY_DEFAULT = CROWDING_DISTANCE_NAME

NONE_NAME = "none"
UNIFORM_NAME = "uniform"
ANOVA_NAME = "anova"

DEFAULT_OBJECTIVE_NAMES = [Leanness().nick(), BalancedAccuracy().nick()]

DEFAULT_DATASET_NAME = BrcaInputCreator().nick()
VIEW_NAMES = [MIRNA_NAME, MRNA_NAME, RPPAA_NAME]
DEFAULT_VIEWS = [MRNA_NAME]

INITIAL_FEATURES_UNIFORM_NAME = "uniform"
INITIAL_FEATURES_BINOMIAL_NAME = "binomial"
INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME = "binomial_from_uniform"
DEFAULT_INITIAL_FEATURES_STRATEGY_NAME = INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME

STRATIFIED_K_FOLD_NAME = "stratified_k_fold"
AUTO_FOLD_NAME = "auto"
DEFAULT_OUTER_FOLDS_NAME = STRATIFIED_K_FOLD_NAME
