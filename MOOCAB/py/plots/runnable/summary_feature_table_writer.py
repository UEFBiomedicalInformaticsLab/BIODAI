from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame

from hall_of_fame.fronts import PARETO_NICK
from input_data.view_prefix import n_views, all_unprefixed
from plots.archives.test_batteries_archive import ALL_BATTERIES
from plots.archives.test_battery_cv import TestBatteryCV
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR
from plots.saved_hof import SavedHoF
from plots.solution_utils import solutions_and_hof_names, solutions_all_folds_and_hof_names, obj_nicks_from_hofs
from saved_solutions.saved_solution import SavedSolution
from util.dataframes import to_csv_makingdirs


MAX_TABLE_CELLS_REASONABLE = 10000000
MAX_TABLE_CELLS_HUGE = 20000000  # Should stay under 200 Mb of csv file.


def features_in_final_solutions(hofs: Sequence[SavedHoF]) -> list[str]:
    solutions, hof_names = solutions_and_hof_names(hofs=hofs)
    features = set()
    for alg_name, alg_solutions in zip(hof_names, solutions):
        for solution in alg_solutions:
            for f in solution.features():
                features.add(f)
    res = list(features)
    res.sort()
    return res


def features_in_all_folds_solutions(hofs: Sequence[SavedHoF]) -> list[str]:
    solutions, hof_names = solutions_all_folds_and_hof_names(hofs=hofs)
    features = set()
    for alg_name, alg_solutions in zip(hof_names, solutions):
        for fold in alg_solutions:
            for solution in fold:
                for f in solution.features():
                    features.add(f)
    res = list(features)
    res.sort()
    return res


def summary_feature_table_writer_one_hof(
        test_battery: TestBatteryCV,
        main_plot_dir: str = SUMMARY_STAT_DIR,
        hof_nick: str = PARETO_NICK,
        max_table_cells: Optional[int] = 10000000):

    for dataset_label in test_battery.dataset_labels():

        print("\nProcessing dataset " + str(dataset_label) + " with hall of fame " + hof_nick)
        hofs = test_battery.flat_hofs_for_dataset(
            dataset_lab=dataset_label, hof_nick=hof_nick)
        dataset_and_objectives_path_part = test_battery.dataset_report_path_part(dataset_lab=dataset_label)
        obj_nicks = obj_nicks_from_hofs(hofs=hofs)

        print("Creating table for final optimization")
        res_columns = {}
        all_features = features_in_final_solutions(hofs=hofs)
        for f in all_features:
            res_columns[f] = []
        row_names = []
        solutions, hof_names = solutions_and_hof_names(hofs=hofs)
        for alg_name, alg_solutions in zip(hof_names, solutions):
            for solution in alg_solutions:
                row_names.append(alg_name)
                s_features = solution.features()
                for f in all_features:
                    if f in s_features:
                        res_columns[f].append("1")
                    else:
                        res_columns[f].append("0")
        res_df = DataFrame(res_columns, index=row_names)
        colnames = res_df.columns
        if n_views(names=colnames) < 2:
            res_df.set_axis(all_unprefixed(names=colnames), axis=1, inplace=True)
        print("Table shape: " + str(res_df.shape))
        if max_table_cells is not None and res_df.shape[0]*res_df.shape[1] > max_table_cells:
            print("Not saving because table is too big.")
        else:
            plot_path = main_plot_dir + "/cv/" + dataset_and_objectives_path_part + "/" + "summary_feature_table_" +\
                        str(hof_nick) + ".csv"
            print("Plotting summary feature table to path " + str(plot_path))
            to_csv_makingdirs(df=res_df, path=plot_path, index=True)

        print("Creating table for fold solutions")
        res_columns = {}
        meta_columns = {}
        all_features = features_in_all_folds_solutions(hofs=hofs)
        res_columns["Optimizer"] = []
        res_columns["Fold"] = []
        meta_columns["Optimizer"] = []
        meta_columns["Fold"] = []
        if obj_nicks is not None:
            for o in obj_nicks:
                for t in ["train", "test"]:
                    meta_columns[o + "_" + t + "_fitness"] = []
                    meta_columns[o + "_" + t + "_std_dev"] = []
                    meta_columns[o + "_" + t + "_ci_min"] = []
                    meta_columns[o + "_" + t + "_ci_max"] = []
        for f in all_features:
            res_columns[f] = []
        solutions, hof_names = solutions_all_folds_and_hof_names(hofs=hofs)
        for alg_name, alg_solutions in zip(hof_names, solutions):
            for i, fold in enumerate(alg_solutions):
                for solution in fold:
                    if not isinstance(solution, SavedSolution):
                        raise ValueError()
                    res_columns["Optimizer"].append(alg_name)
                    res_columns["Fold"].append(i)
                    meta_columns["Optimizer"].append(alg_name)
                    meta_columns["Fold"].append(i)
                    s_features = solution.features()
                    for f in all_features:
                        if f in s_features:
                            res_columns[f].append("1")
                        else:
                            res_columns[f].append("0")
                    if obj_nicks is not None:
                        for j, o in enumerate(obj_nicks):
                            meta_columns[o + "_" + "train" + "_fitness"].append(solution.train_fitnesses()[j])
                            meta_columns[o + "_" + "train" + "_std_dev"].append(solution.train_std_devs()[j])
                            meta_columns[o + "_" + "train" + "_ci_min"].append(solution.train_ci()[j].a())
                            meta_columns[o + "_" + "train" + "_ci_max"].append(solution.train_ci()[j].b())
                            meta_columns[o + "_" + "test" + "_fitness"].append(solution.test_fitnesses()[j])
                            meta_columns[o + "_" + "test" + "_std_dev"].append(solution.test_std_devs()[j])
                            meta_columns[o + "_" + "test" + "_ci_min"].append(solution.test_ci()[j].a())
                            meta_columns[o + "_" + "test" + "_ci_max"].append(solution.test_ci()[j].b())
        res_df = DataFrame(res_columns)
        meta_df = DataFrame(meta_columns)
        if n_views(names=all_features) < 2:
            colnames = ["Optimizer", "Fold"]
            colnames.extend(all_unprefixed(all_features))
            res_df.set_axis(colnames, axis=1, inplace=True)

        print("Features table shape: " + str(res_df.shape))
        if max_table_cells is not None and res_df.shape[0] * res_df.shape[1] > max_table_cells:
            print("Not saving because table is too big.")
        else:
            plot_path = main_plot_dir + "/cv/" + dataset_and_objectives_path_part + "/" + "summary_feature_table_" + \
                        str(hof_nick) + "_folds" + ".csv"
            print("Plotting summary feature table to path " + str(plot_path))
            to_csv_makingdirs(df=res_df, path=plot_path, index=False)

        print("Meta table shape: " + str(meta_df.shape))
        if max_table_cells is not None and meta_df.shape[0] * meta_df.shape[1] > max_table_cells:
            print("Not saving because table is too big.")
        else:
            plot_path = main_plot_dir + "/cv/" + dataset_and_objectives_path_part + "/" + "meta_table_" + \
                        str(hof_nick) + "_folds" + ".csv"
            print("Plotting meta table to path " + str(plot_path))
            to_csv_makingdirs(df=meta_df, path=plot_path, index=False)


def summary_feature_table_writer(
        test_battery: TestBatteryCV,
        main_plot_dir: str = SUMMARY_STAT_DIR,
        hof_nicks: Sequence[str] = (PARETO_NICK,),
        max_table_cells: Optional[int] = MAX_TABLE_CELLS_REASONABLE):
    """Set max_table_cells to None to write tables of any size."""
    for hof_nick in hof_nicks:
        summary_feature_table_writer_one_hof(test_battery=test_battery,
                                             main_plot_dir=main_plot_dir,
                                             hof_nick=hof_nick,
                                             max_table_cells=max_table_cells)


if __name__ == '__main__':

    for battery in ALL_BATTERIES:
        summary_feature_table_writer(test_battery=battery)
