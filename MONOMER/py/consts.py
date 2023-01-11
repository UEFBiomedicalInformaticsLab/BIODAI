from model.model_with_coef import NBWithFallback

DEFAULT_RECURSION_LIMIT = 100000  # At least 20000 needed when pickling big objects.
DEFAULT_MODEL = NBWithFallback()
GEN_STR = "gen"
FOLD_RES_DATA_PREFIX = "log_fold_"
FOLD_RES_DATA_EXTENSION = "csv"
GENERATION_FEATURES_STRING = "gen_feats"
EXPLORED_FEATURES_STRING = "explored"
FEATURE_COUNTS_PREFIX = "feature_counts_"
FEATURE_COUNTS_EXTENSION = "csv"
FINAL_STR = "final"
DEFAULT_FOLD_PARALLELISM = True
MAX_WORKERS = 256
FONT_SIZE = 11
