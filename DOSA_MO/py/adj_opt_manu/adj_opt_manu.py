from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from load_omics_views import MRNA_NAME
from model.regression.pruned_tree import PrunedTree
from model.regression.randomized_svr import RandomizedSVRModel
from model.regression.regressors_library import ZeroRegressor, DummyRegressorModel, RFRegressor, SVRegressor
from objective.objective_with_importance.leanness import RootLeanness
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.archives.test_battery_cv import TestBatteryCV
from plots.plot_labels import TCGA_KID_IHC_DET_LAB, TCGA_KID_IHC_OS_LAB

ADJ_OPT_MANU_REGRESSORS =\
    (ZeroRegressor(), DummyRegressorModel(strategy="median"),
     PrunedTree(square_error=False),
     RFRegressor(criterion="absolute_error"),
     SVRegressor(), RandomizedSVRModel())
ADJ_OPT_MANU_NICK_TO_REGRESSOR = {}
for r in ADJ_OPT_MANU_REGRESSORS:
    ADJ_OPT_MANU_NICK_TO_REGRESSOR[r.nick()] = r

ADJ_OPT_MANU_REGRESSORS_LABS = list(ADJ_OPT_MANU_NICK_TO_REGRESSOR.keys())

ADJ_MANU_TRAIL = "_adj_manu"

TCGA_KID_IHC_DET_ACC_ADJ_MANU_BATTERY = TestBatteryCV(
    objective_computers=[BalancedAccuracy(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_DET_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_KID_IHC_DET_LAB + ADJ_MANU_TRAIL,
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=ADJ_OPT_MANU_REGRESSORS_LABS)

TCGA_KID_IHC_DET_OS_ADJ_MANU_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=3,
    cv_repeats=3,
    dataset_labels=[TCGA_KID_IHC_OS_LAB],
    views=(MRNA_NAME,),
    cox_fi=True,
    nick=TCGA_KID_IHC_OS_LAB + ADJ_MANU_TRAIL,
    generations=GenerationsStrategy(concatenated=500),
    adjuster_regressors=ADJ_OPT_MANU_REGRESSORS_LABS)
