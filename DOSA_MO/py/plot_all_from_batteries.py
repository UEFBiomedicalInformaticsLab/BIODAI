import sys

from consts import DEFAULT_RECURSION_LIMIT
from plots.archives.test_batteries_archive import ALL_BATTERIES
from plots.postprocessing_battery import postprocessing_battery
from plots.runnable.best_genes_plotter_from_batteries import best_genes_plotter
from plots.runnable.performance_gap_analysis_from_registries import performance_gap_analysis_from_battery
from plots.runnable.subplots_by_inner_model_from_batteries import subplots_for_cv_battery_all_pairs
from plots.runnable.summary_statistics_plotter_from_registries import summary_statistics_plotter_from_registries

BATTERIES = ALL_BATTERIES
RUN_POSTPROCESSING = False

if __name__ == '__main__':

    sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)

    for battery in BATTERIES:
        print("\nProcessing test battery " + battery.name())
        if RUN_POSTPROCESSING:
            postprocessing_battery(test_battery=battery)
        performance_gap_analysis_from_battery(test_battery=battery)
        summary_statistics_plotter_from_registries(test_battery=battery)
        subplots_for_cv_battery_all_pairs(test_battery=battery)
        best_genes_plotter(test_battery=battery)
