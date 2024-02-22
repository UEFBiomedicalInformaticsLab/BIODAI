from collections.abc import Sequence

from plots.archives.test_batteries_archive import ALL_BATTERIES
from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.barplot import barplot_with_std_to_file
from plots.default_labels_map import DEFAULT_LABELS_TRANSFORMER
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR
from plots.saved_hof import SavedHoF
from validation_registry.registry_property import RegistryProperty
from validation_registry.registry_property_archive import TEST_HV_PROPERTY, CROSS_HV_PROPERTY, MEAN_JACCARD_PROPERTY, \
    STABILITY_BY_WEIGHTS_PROPERTY, STABILITY_BY_DICE_PROPERTY, STABILITY_BY_BEST_DICE_PROPERTY, INNER_CV_HV_PROPERTY, \
    PERFORMANCE_GAP_PROPERTY, PERFORMANCE_ERROR_PROPERTY, PARETO_DELTA_PROPERTY

DEFAULT_REGISTRY_PROPERTIES = (
    TEST_HV_PROPERTY, CROSS_HV_PROPERTY, INNER_CV_HV_PROPERTY, MEAN_JACCARD_PROPERTY, STABILITY_BY_WEIGHTS_PROPERTY,
    STABILITY_BY_DICE_PROPERTY, STABILITY_BY_BEST_DICE_PROPERTY, PERFORMANCE_GAP_PROPERTY, PERFORMANCE_ERROR_PROPERTY,
    PARETO_DELTA_PROPERTY)


def process_property(
        pre_measure_path: str,
        hofs: Sequence[SavedHoF],
        prop: RegistryProperty):
    measure_nick = prop.nick()
    plot_path = pre_measure_path + measure_nick
    print("Plotting measure " + prop.name() + " to path " + str(plot_path))
    hof_names = []
    classes = []
    measure_vals = []
    for h in hofs:
        registry = h.validation_registry()
        if prop.has_folds_property_name():
            if not prop.has_folds_property(registry=registry):
                folds_prop_name = prop.folds_property_name()
                print("Property " + folds_prop_name + " not found for hall of fame " + h.name() + ".")
                print("Missing property will be computed and saved.")
            measure = prop.smart_extract_folds(hof_path=h.path(), verbose=True)
        else:
            if not prop.has_property(registry=registry):
                prop_name = prop.property_name()
                print("Property " + prop_name + " not found for hall of fame " + h.name() + ".")
                if prop.can_compute():
                    print("Missing property will be computed and saved.")
                    prop.smart_extract(hof_path=h.path(), verbose=True)
                else:
                    print("No computing algorithm is registered for this property.")
            measure = [prop.extract_or_none(registry)]
        if measure is not None:
            measure_vals.append(measure)
            hof_names.append(h.name())
            classes.append(h.main_algorithm_label())
        else:
            print("property " + prop.name() + " not found for " + h.name())
    barplot_with_std_to_file(
        path=plot_path, measures=measure_vals, bar_names=hof_names, label_y=prop.name(), classes=classes,
        labels_transformer=DEFAULT_LABELS_TRANSFORMER)


def summary_statistics_plotter_from_registries(
        test_battery: TestBattery,
        main_plot_dir: str = SUMMARY_STAT_DIR,
        properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES):
    """Properties are processed in sequence order."""
    if test_battery.is_external():
        if not isinstance(test_battery, TestBatteryExternal):
            raise ValueError()
        internal_label = test_battery.internal_dataset_label()
        external_label = test_battery.external_dataset_label()
        print("Processing external validation " + str(internal_label) + " - " + str(external_label))
        hofs = test_battery.existing_flat_hofs()
        dataset_report_path_part = test_battery.dataset_report_path_part()
        pre_measure_path = main_plot_dir + "/external/" + dataset_report_path_part + "/"
        for prop in properties:
            process_property(pre_measure_path=pre_measure_path, hofs=hofs, prop=prop)
    else:
        if not isinstance(test_battery, TestBatteryCV):
            raise ValueError()
        for dataset_label in test_battery.dataset_labels():
            print("\nProcessing dataset " + str(dataset_label))
            hofs = test_battery.existing_flat_hofs_for_dataset(dataset_lab=dataset_label)
            dataset_report_path_part = test_battery.dataset_report_path_part(dataset_lab=dataset_label)
            pre_measure_path = main_plot_dir + "/cv/" + dataset_report_path_part + "/"
            for prop in properties:
                process_property(pre_measure_path=pre_measure_path, hofs=hofs, prop=prop)


if __name__ == '__main__':

    for battery in ALL_BATTERIES:
        print("\nProcessing test battery " + battery.name())
        summary_statistics_plotter_from_registries(test_battery=battery)
