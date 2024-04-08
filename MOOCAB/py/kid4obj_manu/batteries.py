from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from objective.objective_with_importance.leanness import RootLeanness
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.separation.root_separation import RootSeparation
from plots.archives.test_batteries_archive import TCGA_KID_IHC_DET_ACC_NONADJ_BATTERY, \
    TCGA_KID_IHC_DET_OS_ACC_NONADJ_BATTERY, \
    TCGA_KID_IHC_DET_OS_SEP_BATTERY, TCGA_KID_IHC_DET_SEP_BATTERY, TCGA_KID_IHC_DET_ACC_SEP_BATTERY, \
    TCGA_KID_IHC_DET_OS_NONADJ_BATTERY, TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY
from plots.archives.test_battery_cv import TestBatteryCV
from plots.plot_labels import TCGA_KID_IHC_DET_LAB, NSGA3_LAB, NSGA3_CHS_LAB, NB_LAB

TCGA_KID_IHC_DET_ACC_SEP_NB_ONLY_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness(), RootSeparation()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nb_only",
    inner_labs=[NB_LAB],
    generations=GenerationsStrategy(concatenated=500))


TCGA_KID_IHC_DET_ACC_SEP_NB_ONLY_NOCHP_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness(), RootSeparation()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    main_labs=[NSGA3_LAB, NSGA3_CHS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nb_only_nochp",
    inner_labs=[NB_LAB],
    generations=GenerationsStrategy(concatenated=500))


TCGA_KID_IHC_DET_ACC_NONADJ_NB_ONLY_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nonadj" + "_nb_only",
    generations=GenerationsStrategy(concatenated=500),
    inner_labs=[NB_LAB],
    adjuster_regressors=[None])


TCGA_KID_IHC_DET_ACC_NONADJ_NB_ONLY_NOCHP_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    main_labs=[NSGA3_LAB, NSGA3_CHS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nonadj" + "_nb_only_nochp",
    generations=GenerationsStrategy(concatenated=500),
    inner_labs=[NB_LAB],
    adjuster_regressors=[None])


TCGA_KID_IHC_DET_SEP_NOCHP_BATTERY = TestBatteryCV(
    objective_computers=[RootSeparation(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    main_labs=[NSGA3_LAB, NSGA3_CHS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nochp",
    generations=GenerationsStrategy(concatenated=500))


TCGA_KID_IHC_DET_ACC_SEP_NOCHP_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness(), RootSeparation()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    main_labs=[NSGA3_LAB, NSGA3_CHS_LAB],
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + "_nochp",
    generations=GenerationsStrategy(concatenated=500))


KID4OBJ_NB_ONLY_NOCHP_BATTERIES = [
    TCGA_KID_IHC_DET_ACC_NONADJ_NB_ONLY_NOCHP_BATTERY,
    TCGA_KID_IHC_DET_ACC_SEP_NB_ONLY_NOCHP_BATTERY,
    TCGA_KID_IHC_DET_SEP_NOCHP_BATTERY
]


KID4OBJ_BATTERIES = [
    TCGA_KID_IHC_DET_ACC_SEP_NB_ONLY_NOCHP_BATTERY,
    TCGA_KID_IHC_DET_ACC_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_ACC_NONADJ_NB_ONLY_BATTERY,
    TCGA_KID_IHC_DET_ACC_NONADJ_NB_ONLY_NOCHP_BATTERY,
    TCGA_KID_IHC_DET_OS_ACC_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_OS_SEP_BATTERY,
    TCGA_KID_IHC_DET_SEP_BATTERY,
    TCGA_KID_IHC_DET_SEP_NOCHP_BATTERY,
    TCGA_KID_IHC_DET_ACC_SEP_BATTERY,
    TCGA_KID_IHC_DET_ACC_SEP_NB_ONLY_BATTERY,
    TCGA_KID_IHC_DET_OS_NONADJ_BATTERY,
    TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY
]
