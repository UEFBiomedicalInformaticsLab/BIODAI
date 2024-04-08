import sys

from consts import DEFAULT_RECURSION_LIMIT
from hall_of_fame.hof_by_sum import HofBySum
from hall_of_fame.fronts import PARETO_NICK
from kid4obj_manu.batteries import KID4OBJ_BATTERIES
from plots.postprocessing_battery import postprocessing_battery
from plots.runnable.best_genes_plotter_from_batteries import best_features_plotter
from plots.runnable.subplots_by_inner_model_from_batteries import subplots_for_cv_battery_all_pairs
from plots.runnable.summary_feature_table_writer import summary_feature_table_writer, MAX_TABLE_CELLS_HUGE
from plots.runnable.summary_statistics_plotter_from_registries import summary_statistics_plotter_from_registries

BATTERIES = KID4OBJ_BATTERIES
RUN_POSTPROCESSING = False
CREATE_SUMMARY_FEATURE_TABLES = False
RUN_DETAILED_PLOTS = True
MAX_TABLE_CELLS = MAX_TABLE_CELLS_HUGE  # None to have no maximum.


if __name__ == '__main__':

    sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)

    for battery in BATTERIES:
        print("\nProcessing test battery " + battery.name())
        if RUN_POSTPROCESSING:
            postprocessing_battery(test_battery=battery)
        if RUN_DETAILED_PLOTS:
            subplots_for_cv_battery_all_pairs(test_battery=battery)
        if CREATE_SUMMARY_FEATURE_TABLES:
            summary_feature_table_writer(test_battery=battery,
                                         hof_nicks=(PARETO_NICK, HofBySum(size=100).nick()),
                                         max_table_cells=MAX_TABLE_CELLS)
        best_features_plotter(test_battery=battery)
        summary_statistics_plotter_from_registries(test_battery=battery)
