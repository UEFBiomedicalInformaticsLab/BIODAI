from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from load_omics_views import MRNA_NAME
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import RootLeanness
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.plot_labels import (TCGA_BRCA_LAB, TCGA_KID_IHC_DET_LAB, TCGA_KID_IHC_OS_LAB,
                               PROPER_ADJUSTER_REGRESSORS_LABS, SWEDISH_LAB)


TCGA_KID_IHC_DET_ACC_ADJ_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_adj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS)

TCGA_KID_IHC_DET_OS_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB,
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS)

TCGA_KID_IHC_DET_OS_ACC_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB,
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS)

TCGA_BRCA_MRNA_ADJ_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[TCGA_BRCA_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_BRCA_LAB + "_adj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS)

TCGA_BRCA_SWEDISH_MRNA_ADJ_ACC_BATTERY = TestBatteryExternal(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    internal_dataset_label=TCGA_BRCA_LAB,
    external_dataset_label=SWEDISH_LAB,
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_BRCA_LAB + "_" + SWEDISH_LAB + "_adj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS
    )

TCGA_BRCA_MRNA_OS_ADJ_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[TCGA_BRCA_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_BRCA_LAB + "_adj",
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=PROPER_ADJUSTER_REGRESSORS_LABS
    )

ALL_BATTERIES = [
    TCGA_KID_IHC_DET_ACC_ADJ_BATTERY,
    TCGA_KID_IHC_DET_OS_ACC_BATTERY,
    TCGA_KID_IHC_DET_OS_BATTERY,
    TCGA_BRCA_MRNA_ADJ_BATTERY,
    TCGA_BRCA_SWEDISH_MRNA_ADJ_ACC_BATTERY,
    TCGA_BRCA_MRNA_OS_ADJ_BATTERY
    ]
