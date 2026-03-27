from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from load_omics_views import CLINIC_NAME, LOG_MRNA_NAME, LOG_MIRNA_NAME
from manu.mv_manu.mv_manu_consts import MV_MANU_PLOT_SETUP
from objective.objective_with_importance.leanness import RootLeanness
from objective.objective_with_importance.survival_objective_computer_with_importance import CIndex
from plots.archives.test_batteries_archive import all_view_combinations, MV_BASELINE, MV_GENERATIONS
from plots.archives.test_battery_cv import TestBatteryCV
from plots.plot_labels import KIRC_MV_LAB, COAD_MV_LAB, SARC_MV_LAB, LGG_MV_LAB


MV_MANU_VIEWS = [CLINIC_NAME, LOG_MRNA_NAME, LOG_MIRNA_NAME]

KIRC_MV_MANU_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[KIRC_MV_LAB],
    view_defs=all_view_combinations(
        included_views=MV_MANU_VIEWS),
    generations=[GenerationsStrategy(concatenated=300), GenerationsStrategy(sweeps=[150]),
                 GenerationsStrategy(sweeps=[50, 50, 50]), GenerationsStrategy(sweeps=[50, 50], concatenated=100)],
    population=[500],
    adjuster_regressors=[None],
    nick="kirc_mv_manu",
    baseline=MV_BASELINE,
    plot_setup=MV_MANU_PLOT_SETUP)

COAD_MV_MANU_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[COAD_MV_LAB],
    view_defs=all_view_combinations(
        included_views=MV_MANU_VIEWS),
    generations=[GenerationsStrategy(concatenated=300), GenerationsStrategy(sweeps=[150]),
                 GenerationsStrategy(sweeps=[50, 50, 50]), GenerationsStrategy(sweeps=[50, 50], concatenated=100)],
    population=[500],
    adjuster_regressors=[None],
    nick="coad_mv_manu",
    baseline=MV_BASELINE,
    plot_setup=MV_MANU_PLOT_SETUP
)

SARC_MV_MANU_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[SARC_MV_LAB],
    view_defs=all_view_combinations(
        included_views=MV_MANU_VIEWS),
    generations=[GenerationsStrategy(concatenated=300), GenerationsStrategy(sweeps=[150]),
                 GenerationsStrategy(sweeps=[50, 50, 50]), GenerationsStrategy(sweeps=[50, 50], concatenated=100)],
    population=[500],
    adjuster_regressors=[None],
    nick="sarc_mv_manu",
    baseline=MV_BASELINE,
    plot_setup=MV_MANU_PLOT_SETUP)

LGG_MV_MANU_BATTERY = TestBatteryCV(
    objective_computers=[CIndex(), RootLeanness()],
    n_outer_folds=5,
    dataset_labels=[LGG_MV_LAB],
    view_defs=all_view_combinations(
        included_views=MV_MANU_VIEWS),
    generations=MV_GENERATIONS,
    population=[500],
    adjuster_regressors=[None],
    nick="lgg_mv_manu",
    baseline=MV_BASELINE,
    plot_setup=MV_MANU_PLOT_SETUP
)

MV_MANU_HOFS = []
MV_MANU_HOFS.extend(KIRC_MV_MANU_BATTERY.existing_flat_hofs())
MV_MANU_HOFS.extend(LGG_MV_MANU_BATTERY.existing_flat_hofs())
MV_MANU_HOFS.extend(SARC_MV_MANU_BATTERY.existing_flat_hofs())
