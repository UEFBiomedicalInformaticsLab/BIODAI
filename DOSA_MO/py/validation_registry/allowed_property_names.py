from __future__ import annotations

from util.cross_hypervolume.pareto_delta import PARETO_DELTA_NAME

INNER_CV_HV_NICK = "inner_cv_hypervol"
TEST_HV_NICK = "test_hypervol"
CROSS_HV_NICK = "cross_hypervol"
PERFORMANCE_GAP_NICK = "perf_gap"
PERFORMANCE_ERROR_NICK = "perf_err"
MEAN_JACCARD_NICK = "mean_Jaccard"
STABILITY_BY_WEIGHTS_NICK = "weight_overlap"
STABILITY_BY_DICE_NICK = "Dice"  # Same as stability by unions
STABILITY_BY_BEST_DICE_NICK = "best_dice"

INNER_CV_HV_NAME = "inner CV hypervolume"
TRAIN_HV_NAME = "train hypervolume"
TEST_HV_NAME = "test hypervolume"
CROSS_HV_NAME = "cross hypervolume"
PERFORMANCE_GAP_NAME = "performance gap"
PERFORMANCE_ERROR_NAME = "performance error"
FOLDS_PREFIX = "folds "
FOLDS_INNER_CV_HV_NAME = FOLDS_PREFIX + "inner CV hypervolume"
FOLDS_TRAIN_HV_NAME = FOLDS_PREFIX + "train hypervolume"
FOLDS_TEST_HV_NAME = FOLDS_PREFIX + "test hypervolume"
FOLDS_CROSS_HV_NAME = FOLDS_PREFIX + "cross hypervolume"
FOLDS_PERFORMANCE_GAP_NAME = FOLDS_PREFIX + "performance gap"
FOLDS_PERFORMANCE_ERROR_NAME = FOLDS_PREFIX + "performance error"
FOLDS_PARETO_DELTA_NAME = FOLDS_PREFIX + PARETO_DELTA_NAME
MEAN_JACCARD_NAME = "mean Jaccard"
FOLDS_MEAN_JACCARD_NAME = FOLDS_PREFIX + "mean Jaccard"
STABILITY_BY_WEIGHTS_NAME = "stability by weight overlap"
STABILITY_BY_DICE_NAME = "stability by Dice"  # Same as stability by unions
STABILITY_BY_BEST_DICE_NAME = "stability by best Dice"
STABILITY_BY_SPEARMAN_NAME = "stability by Spearman"
STABILITY_BY_TOP50_NAME = "stability by top50"
STABILITY_OF_VIEWS_NAME = "stability of views"
DISTRIBUTION_OF_VIEWS_NAME = "distribution of views"

ALLOWED_PROPERTY_NAMES = {
    INNER_CV_HV_NAME, TRAIN_HV_NAME, TEST_HV_NAME, CROSS_HV_NAME,
    FOLDS_INNER_CV_HV_NAME, FOLDS_TRAIN_HV_NAME, FOLDS_TEST_HV_NAME, FOLDS_CROSS_HV_NAME,
    MEAN_JACCARD_NAME, FOLDS_MEAN_JACCARD_NAME,
    STABILITY_BY_WEIGHTS_NAME, STABILITY_BY_DICE_NAME, STABILITY_BY_BEST_DICE_NAME,
    STABILITY_BY_SPEARMAN_NAME, STABILITY_BY_TOP50_NAME,
    STABILITY_OF_VIEWS_NAME, DISTRIBUTION_OF_VIEWS_NAME,
    PERFORMANCE_GAP_NAME, FOLDS_PERFORMANCE_GAP_NAME,
    PERFORMANCE_ERROR_NAME, FOLDS_PERFORMANCE_ERROR_NAME,
    PARETO_DELTA_NAME, FOLDS_PARETO_DELTA_NAME
}

RENAMES = [("stability by weights", STABILITY_BY_WEIGHTS_NAME),
           ("stability by unions", STABILITY_BY_DICE_NAME)]
