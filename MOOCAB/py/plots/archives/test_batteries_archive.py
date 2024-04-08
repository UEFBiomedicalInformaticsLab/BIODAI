from collections.abc import Iterable, Sequence

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import RootLeanness
from objective.objective_with_importance.separation.root_separation import RootSeparation
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.archives.test_battery_cv import TestBatteryCV
from plots.plot_labels import TCGA_KID_IHC_DET_LAB, TCGA_KID_IHC_OS_LAB
from util.math.list_math import powerset


def all_view_combinations(included_views: Iterable[str]) -> Sequence[set[str]]:
    return list(powerset(iterable=included_views, include_empty=False))


TCGA_KID_IHC_DET_ACC_SEP_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness(), RootSeparation()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB,
    generations=GenerationsStrategy(concatenated=500))

TCGA_KID_IHC_DET_ACC_NONADJ_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nonadj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=[None])

TCGA_KID_IHC_DET_SEP_BATTERY = TestBatteryCV(
    objective_computers=[RootSeparation(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB,
    generations=GenerationsStrategy(concatenated=500))

TCGA_KID_IHC_DET_OS_NONADJ_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB + "_nonadj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=[None])

TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), BalancedAccuracy(), RootLeanness(), RootSeparation()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB,
    generations=GenerationsStrategy(concatenated=500))

TCGA_KID_IHC_DET_OS_SEP_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootSeparation(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB,
    generations=GenerationsStrategy(concatenated=500))

TCGA_KID_IHC_DET_OS_ACC_NONADJ_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB + "_nonadj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=[None])

ALL_BATTERIES = [
    TCGA_KID_IHC_DET_ACC_SEP_BATTERY,
    TCGA_KID_IHC_DET_ACC_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_SEP_BATTERY,
    TCGA_KID_IHC_DET_OS_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_OS_SEP_BATTERY,
    TCGA_KID_IHC_DET_OS_ACC_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY]
