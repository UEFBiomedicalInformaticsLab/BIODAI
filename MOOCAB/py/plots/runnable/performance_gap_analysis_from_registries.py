from collections.abc import Sequence

from pandas import DataFrame

from plots.archives.test_batteries_archive import ALL_BATTERIES
from plots.archives.test_battery_cv import TestBatteryCV
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR
from util.dataframes import to_csv_makingdirs
from util.math.sequences_to_float import PearsonCorr, SequencesToFloat, SpearmanCorr, KendallCorr

GAP_ANALYSIS_STR = "gap_analysis"
DEFAULT_MEASURES = (PearsonCorr(), PearsonCorr(p_val=True),
                    SpearmanCorr(), SpearmanCorr(p_val=True),
                    KendallCorr(), KendallCorr(p_val=True))


def performance_gap_analysis_from_battery(
        test_battery: TestBatteryCV,
        measures: Sequence[SequencesToFloat] = DEFAULT_MEASURES,
        main_plot_dir: str = SUMMARY_STAT_DIR,
        verbose: bool = False):
    """Correlations and p-values between standard deviation and performance gap."""

    if test_battery.is_external():
        print("Skipping performance gap analysis for external test battery " + test_battery.nick())
    else:
        for dataset_label in test_battery.dataset_labels():
            print("\nProcessing dataset " + str(dataset_label))
            hofs = test_battery.existing_flat_hofs_for_dataset(dataset_lab=dataset_label)
            dataset_and_objectives_path_part = test_battery.dataset_report_path_part(dataset_lab=dataset_label)
            for measure in measures:
                res_dict = {"Optimizer": [], "Objective": []}
                n_fold_columns = test_battery.cv_repeats()*test_battery.n_outer_folds()
                for i in range(n_fold_columns):
                    res_dict["Fold" + str(i)] = []
                for hof in hofs:
                    hof_name = hof.name()
                    if hof.has_obj_nicks():
                        obj_nicks = hof.obj_nicks()
                    else:
                        obj_nicks = []
                    measures_by_fold = hof.sd_gap_measures_by_fold(measure=measure)
                    if measures_by_fold is not None:
                        if verbose:
                            print(hof_name)
                            print("Measures: " + str(measures_by_fold))
                        if len(measures_by_fold) < n_fold_columns:
                            print("Not enough saved folds: skipping.")
                        else:
                            for nick_index in range(len(obj_nicks)):
                                res_dict["Optimizer"].append(hof_name)
                                res_dict["Objective"].append(obj_nicks[nick_index])
                                for j in range(n_fold_columns):
                                    res_dict["Fold" + str(j)].append(measures_by_fold[j][nick_index])
                if len(res_dict["Optimizer"]) > 0:
                    measure_nick = measure.nick()
                    file_name = "sd_gap_" + measure_nick + ".csv"
                    plot_path = (main_plot_dir + "/cv/" + dataset_and_objectives_path_part + "/" +
                                 GAP_ANALYSIS_STR + "/" + file_name)
                    print("Plotting measure " + measure.name() + " to path " + str(plot_path))
                    res = DataFrame(res_dict)
                    to_csv_makingdirs(df=res, path=plot_path, index=False)


if __name__ == '__main__':

    for battery in ALL_BATTERIES:
        print("\nProcessing test battery " + battery.name())
        performance_gap_analysis_from_battery(test_battery=battery)
