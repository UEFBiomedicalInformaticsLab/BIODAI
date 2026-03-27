from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.runnable.best_genes_plotter import best_features_plotter_process_dataset, BEST_FEATURES_STR
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR


def best_features_plotter(
        test_battery: TestBattery,
        main_plot_dir: str = SUMMARY_STAT_DIR,
        show_counts: bool = False):
    type_str = test_battery.type_str()
    if test_battery.is_external():
        if isinstance(test_battery, TestBatteryExternal):
            for external_validation_datasets in test_battery.datasets():
                print("\nProcessing datasets " + str(external_validation_datasets))
                save_path = (main_plot_dir + "/" + type_str + "/" +
                             test_battery.datasets_report_path_part(datasets=external_validation_datasets) + "/" +
                             BEST_FEATURES_STR)
                print("Saving to " + save_path)
                best_features_plotter_process_dataset(save_path=save_path,
                                                      hofs=test_battery.existing_flat_hofs_for_datasets(
                                                       datasets=external_validation_datasets),
                                                      labels_transformer=test_battery.plot_setup().labels_map(),
                                                      show_counts=show_counts)
        else:
            raise ValueError()
    else:
        if isinstance(test_battery, TestBatteryCV):
            for dataset_label in test_battery.dataset_labels():
                print("\nProcessing dataset " + str(dataset_label))
                save_path = main_plot_dir + "/" + type_str + "/" +\
                    test_battery.dataset_report_path_part(dataset_lab=dataset_label) + "/" + BEST_FEATURES_STR
                print("Saving to " + save_path)
                best_features_plotter_process_dataset(save_path=save_path,
                                                      hofs=test_battery.flat_hofs_for_dataset(dataset_lab=dataset_label),
                                                      labels_transformer=test_battery.plot_setup().labels_map(),
                                                      show_counts=show_counts)
        else:
            raise ValueError()
