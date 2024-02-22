import os
from collections.abc import Sequence

from plots.runnable.summary_statistics_plotter_from_registries import DEFAULT_REGISTRY_PROPERTIES
from plots.saved_hof import validation_registry_from_hof_path
from util.system_utils import subdirectories
from validation_registry.registry_property import RegistryProperty


def fill_missing_properties_one_hof(
        hof_dir: str, properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES,
        include_folds: bool = True):
    for prop in properties:
        if os.path.isdir(hof_dir):
            registry = validation_registry_from_hof_path(hof_path=hof_dir)
            if include_folds and prop.has_folds_property_name():
                if not prop.has_folds_property(registry=registry):
                    folds_prop_name = prop.folds_property_name()
                    print("Property " + folds_prop_name + " not found for hall of fame.")
                    if prop.can_compute_for_folds():
                        print("Missing property will be computed and saved.")
                        try:
                            prop.smart_extract_folds(hof_path=hof_dir, verbose=True)
                        except BaseException as e:
                            print("Computing property " + folds_prop_name + " failed with the following exception.\n" +
                                  str(e) + "\n" +
                                  "The program will try to continue.")
                    else:
                        print("No computing algorithm is registered for this property.")
            if not prop.has_property(registry=registry):
                prop_name = prop.property_name()
                print("Property " + prop_name + " not found for hall of fame.")
                if prop.can_compute():
                    print("Missing property will be computed and saved.")
                    try:
                        prop.smart_extract(hof_path=hof_dir, verbose=True)
                    except BaseException as e:
                        print("Computing property " + prop_name + " failed with the following exception.\n" +
                              str(e) + "\n" +
                              "The program will try to continue.")
                else:
                    print("No computing algorithm is registered for this property.")

        else:
            print(str(hof_dir) + " does not exist.")


def fill_missing_properties_every_hof(main_hofs_dir: str, include_folds: bool = True,
                                      properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES):
    for f in subdirectories(main_directory=main_hofs_dir):
        fill_missing_properties_one_hof(f, include_folds=include_folds, properties=properties)
