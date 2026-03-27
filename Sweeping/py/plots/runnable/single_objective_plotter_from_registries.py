from collections.abc import Sequence

from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.barplot import barplot_with_std_to_file
from plots.default_labels_map import LabelsTransformer, DEFAULT_LABELS_TRANSFORMER
from plots.runnable.summary_statistics_plotter_from_registries import pre_measure_path_external, pre_measure_path_cv
from plots.saved_hof import SavedHoF
from survival_plots.survival_summary_statistics_subplotter import SUMMARY_STAT_DIR
from util.math.summer import KahanSummer
from util.sequence_utils import max_positions
from util.name_parts import names_by_differences


def process_objective(
        pre_measure_path: str,
        hofs: Sequence[SavedHoF],
        obj_pos: int,
        labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER):
    if len(hofs) > 0:
        hof_name_parts = []
        classes = []
        measure_vals = []
        obj_nick = "unknown_objective"
        for h in hofs:
            try:
                obj_nick = h.obj_nicks()[obj_pos]
                train_fitness_folds = h.train_fitness_objective_folds(obj = obj_pos)
                test_fitness_folds = h.test_fitness_objective_folds(obj = obj_pos)
                fold_vals = []
                for fold_train, fold_test in zip(train_fitness_folds, test_fitness_folds):
                    max_pos = max_positions(fold_train)
                    test_vals = [fold_test[p] for p in max_pos]
                    fold_vals.append(KahanSummer.mean(test_vals))
                measure_vals.append(fold_vals)
                hof_name_parts.append(labels_transformer.apply_all(h.name_parts()))
                classes.append(h.main_algorithm_label())
            except BaseException as e:
                print("Necessary data is not available. Impossible to plot.")
                print("Original exception:\n" + str(e))
        plot_path = pre_measure_path + obj_nick
        print("Plotting objective " + obj_nick + " to path " + str(plot_path))
        hof_names = names_by_differences(object_features=hof_name_parts)
        barplot_with_std_to_file(
            path=plot_path, measures=measure_vals, bar_names=hof_names, label_y=obj_nick, classes=classes,
            labels_transformer=labels_transformer)
    else:
        print("No hall of fame available. Impossible to plot.")


def single_objective_plotter_from_registries(
        test_battery: TestBattery,
        main_plot_dir: str = SUMMARY_STAT_DIR):
    labels_transformer = test_battery.plot_setup().labels_map()
    if test_battery.is_external():
        if not isinstance(test_battery, TestBatteryExternal):
            raise ValueError()
        for d in test_battery.datasets():
            print("Processing single objectives for external validation " + str(d))
            hofs = test_battery.existing_flat_hofs_for_datasets(datasets=d)
            dataset_report_path_part = test_battery.datasets_report_path_part(datasets=d)
            pre_measure_path = pre_measure_path_external(
                    main_plot_dir=main_plot_dir, dataset_report_path_part=dataset_report_path_part)
            for i in range(test_battery.n_objectives()):
                process_objective(pre_measure_path=pre_measure_path, hofs=hofs, obj_pos=i,
                                 labels_transformer=labels_transformer)
    else:
        if not isinstance(test_battery, TestBatteryCV):
            raise ValueError()
        for dataset_label in test_battery.dataset_labels():
            print("\nProcessing single objectives for dataset " + str(dataset_label))
            hofs = test_battery.existing_flat_hofs_for_dataset(dataset_lab=dataset_label)
            dataset_report_path_part = test_battery.dataset_report_path_part(dataset_lab=dataset_label)
            pre_measure_path = pre_measure_path_cv(
                    main_plot_dir=main_plot_dir, dataset_report_path_part=dataset_report_path_part)
            for i in range(test_battery.n_objectives()):
                process_objective(pre_measure_path=pre_measure_path, hofs=hofs, obj_pos=i,
                                 labels_transformer=labels_transformer)
