import sys
from collections.abc import Iterable, Sequence
from typing import Optional

import matplotlib

from consts import DEFAULT_RECURSION_LIMIT
from hall_of_fame.hof_by_sum import HofBySum
from hall_of_fame.hof_names import PARETO_NICK
from manu.mv_manu.mv_manu_commands import MV_MANU_COMMANDS
from plots.plot_command.plot_command import PlotCommand
from plots.archives.test_batteries_archive import TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY
from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.baseline_best_comparison import baseline_best_comparison_all_pairs
from plots.best_hof_for_dataset_cv import save_best_hof_for_dataset_cv
from plots.postprocessing_battery import postprocessing_battery
from plots.runnable.best_genes_plotter_from_batteries import best_features_plotter
from plots.runnable.best_solutions_printer import save_best_solutions_for_dataset, BEST_SOLUTIONS_STR
from plots.runnable.performance_gap_analysis_from_registries import performance_gap_analysis_from_battery
from plots.runnable.single_objective_plotter_from_registries import single_objective_plotter_from_registries
from plots.runnable.subplots_by_inner_model_from_batteries import subplots_for_battery_all_pairs
from plots.runnable.summary_feature_table_writer import summary_feature_table_writer, MAX_TABLE_CELLS_REASONABLE
from plots.runnable.summary_statistics_plotter_from_registries import summary_statistics_plotter_from_registries, \
    DEFAULT_REGISTRY_PROPERTIES
from survival_plots.survival_summary_statistics_subplotter import SUMMARY_STAT_DIR
from validation_registry.registry_property import RegistryProperty


TCGA_KID_IHC_DET_OS_ACC_SEP_COMMAND = PlotCommand(
    batteries=[TCGA_KID_IHC_DET_OS_ACC_SEP_BATTERY],
    run_postprocessing=False,
    run_detailed_plots=False,
    create_summary_feature_tables=False,
    max_table_cells=None,
    properties=DEFAULT_REGISTRY_PROPERTIES)


# COMMANDS = [TCGA_KID_IHC_DET_OS_ACC_SEP_COMMAND]
# COMMANDS = [LGG_MV_COMMAND]
# COMMANDS = [SARC_MV_COMMAND]
# COMMANDS = [KIRC_MV_COMMAND]
# COMMANDS = [COAD_MV_COMMAND]
# COMMANDS = MV_COMMANDS
COMMANDS = MV_MANU_COMMANDS
# COMMANDS = VINTAGE_ALL_COMMANDS
# COMMANDS = [BRCA_ADJ_COMMAND]
# COMMANDS = [TCGA_KID_IHC_DET_ADJ_COMMAND]
# COMMANDS = [ADJ_OPT_MANU_COMMAND]
# COMMANDS = [TCGA_BRCA_MRNA_SURV_COMMAND]
# COMMANDS = [TCGA_KID_IHC_DET_OS_ACC_SEP_COMMAND, BRCA_ADJ_COMMAND, TCGA_KID_IHC_DET_ADJ_COMMAND, ADJ_OPT_MANU_COMMAND, TCGA_BRCA_MRNA_SURV_COMMAND] + MV_COMMANDS + VINTAGE_ALL_COMMANDS
# COMMANDS = [UKBB_CLINIC_TEST_FS_COMMAND]

BEST_HOF_STR = "best_hof.txt"


def plot_all_from_batteries(
        batteries: Iterable[TestBattery],
        run_postprocessing: bool = False,
        run_detailed_plots: bool = True,
        create_summary_feature_tables: bool = False,
        max_table_cells: Optional[int] = MAX_TABLE_CELLS_REASONABLE,
        properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES,
        show_gene_counts: bool = False):
    """max_table_cells: None to have no maximum."""
    matplotlib.use('Agg')  # Otherwise it might try to go interactive while debugging, then throw exceptions.
    sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)

    for battery in batteries:
        print("\nProcessing test battery " + battery.name())
        if isinstance(battery, TestBatteryCV):
            if run_postprocessing:
                postprocessing_battery(test_battery=battery)
            if create_summary_feature_tables:
                summary_feature_table_writer(test_battery=battery,
                                             hof_nicks=(PARETO_NICK, HofBySum(size=100).nick()),
                                             max_table_cells=max_table_cells)
            performance_gap_analysis_from_battery(test_battery=battery)
            for dataset in battery.dataset_labels():
                best_hof_path = (SUMMARY_STAT_DIR + "/" + battery.type_str() + "/" +
                                 battery.dataset_report_path_part(dataset_lab=dataset) + "/" +
                                 BEST_HOF_STR)
                save_best_hof_for_dataset_cv(
                    save_path=best_hof_path, hofs=battery.existing_flat_hofs_for_dataset(dataset_lab=dataset))
        if run_detailed_plots:
            subplots_for_battery_all_pairs(test_battery=battery)
        best_features_plotter(test_battery=battery, show_counts=show_gene_counts)
        summary_statistics_plotter_from_registries(test_battery=battery, properties=properties)
        single_objective_plotter_from_registries(test_battery=battery)
        baseline_best_comparison_all_pairs(test_battery=battery)
        if isinstance(battery, TestBatteryExternal):
            labels_transformer = battery.plot_setup().labels_map()
            for datasets in battery.datasets():
                hofs = battery.existing_flat_hofs_for_datasets(datasets=datasets)
                print("Processing best biomarkers following external validation of " + str(datasets))
                plot_path = (SUMMARY_STAT_DIR + "/" + battery.type_str() + "/" +
                             battery.datasets_report_path_part(datasets=datasets) + "/" +
                             BEST_SOLUTIONS_STR)
                save_best_solutions_for_dataset(
                    save_path=plot_path, hofs=hofs, labels_transformer=labels_transformer)


def plot_all_from_command(command: PlotCommand):
    plot_all_from_batteries(
        batteries=command.batteries(),
        run_postprocessing=command.run_postprocessing(),
        run_detailed_plots=command.run_detailed_plots(),
        create_summary_feature_tables=command.create_summary_feature_tables(),
        max_table_cells=command.max_table_cells(),
        properties=command.properties(),
        show_gene_counts=command.show_gene_counts()
    )


def plot_all_from_commands(commands: Iterable[PlotCommand]):
    for c in commands:
        plot_all_from_command(command=c)


if __name__ == '__main__':
    plot_all_from_commands(commands=COMMANDS)
