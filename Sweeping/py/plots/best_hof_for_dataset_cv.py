from collections.abc import Sequence
from typing import Optional

from plots.saved_hof import SavedHoF, existing_hofs
from saved_solutions.solutions_from_files import final_solutions_from_files
from util.printer.printer import LogPrinter
from validation_registry.registry_property import RegistryProperty
from validation_registry.registry_property_archive import CROSS_HV_PROPERTY


DEFAULT_COMPARISON_PROPERTY = CROSS_HV_PROPERTY


def best_hof_for_property(hofs: Sequence[SavedHoF],
                          registry_property: RegistryProperty = DEFAULT_COMPARISON_PROPERTY) -> Optional[SavedHoF]:
    """Higher property values are considered better. If there are different seeds, the better performing one will
    be considered."""
    best_hof = None
    best_prop = None
    for h in hofs:
        try:
            prop = registry_property.smart_extract(h.path())
            if best_prop is None or best_prop < prop:
                best_hof = h
                best_prop = prop
        except ZeroDivisionError:
            pass
    return best_hof


def best_hof_for_dataset_str(hofs: Sequence[SavedHoF],
                             registry_property: RegistryProperty = DEFAULT_COMPARISON_PROPERTY) -> str:
    """Higher property values are considered better."""
    if len(hofs) == 0:
        return "No HoFs found."
    best_hof = best_hof_for_property(hofs=hofs, registry_property=registry_property)
    if best_hof is None:
        return "No HoFs found."
    hofers = final_solutions_from_files(hof_dir=best_hof.path())
    res = ""
    res += best_hof.name() + "\n\n"
    if len(hofers) == 0:
        res += "No final optimization hofers found for the best HoF."
        return res
    non_empty = []
    for h in hofers:
        if h.num_features() > 0:
            non_empty.append(h)
    non_empty.sort(key=lambda e: e.train_fitnesses(), reverse=False)
    for h in non_empty:
        res += h.compact_str() + "\n"
    return res


def save_best_hof_for_dataset_cv(save_path: str, hofs: Sequence[SavedHoF],
                                 registry_property: RegistryProperty = DEFAULT_COMPARISON_PROPERTY):
    """Higher property values are considered better."""
    hofs = existing_hofs(hofs=hofs)
    to_write = best_hof_for_dataset_str(hofs=hofs, registry_property=registry_property)
    printer = LogPrinter(log_file=save_path)
    printer.print(to_write)