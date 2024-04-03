from plots.archives.test_batteries_archive import ALL_BATTERIES
from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.runnable.best_genes_plotter import best_genes_plotter_process_dataset, BEST_FEATURES_STR
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR


def best_features_plotter(
        test_battery: TestBattery,
        main_plot_dir: str = SUMMARY_STAT_DIR):
    type_str = test_battery.type_str()
    if test_battery.is_external():
        if isinstance(test_battery, TestBatteryExternal):
            print("\nProcessing datasets " + test_battery.internal_dataset_label() + " - " +
                  test_battery.external_dataset_label())
            save_path = main_plot_dir + "/" + type_str + "/" +\
                test_battery.dataset_report_path_part() + "/" + BEST_FEATURES_STR
            print("Saving to " + save_path)
            best_genes_plotter_process_dataset(save_path=save_path,
                                               hofs=test_battery.flat_hofs())
        else:
            raise ValueError()
    else:
        if isinstance(test_battery, TestBatteryCV):
            for dataset_label in test_battery.dataset_labels():
                print("\nProcessing dataset " + str(dataset_label))
                save_path = main_plot_dir + "/" + type_str + "/" +\
                    test_battery.dataset_report_path_part(dataset_lab=dataset_label) + "/" + BEST_FEATURES_STR
                print("Saving to " + save_path)
                best_genes_plotter_process_dataset(save_path=save_path,
                                                   hofs=test_battery.flat_hofs_for_dataset(dataset_lab=dataset_label))
        else:
            raise ValueError()


if __name__ == '__main__':
    for battery in ALL_BATTERIES:
        print("\nProcessing test battery " + battery.name())
        best_features_plotter(test_battery=battery)
