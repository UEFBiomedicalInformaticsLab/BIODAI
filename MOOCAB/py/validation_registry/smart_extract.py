from __future__ import annotations

from typing import Optional, Sequence, Any

from plots.saved_hof import validation_registry_from_hof_path
from saved_solutions.solutions_from_files import n_folds_from_files
from util.utils import is_sequence_not_string
from validation_registry.hof_property_computer import HofPropertyComputer


def __smart_extract_folds(
        prop_name: str,
        hof_path: str,
        computer: Optional[HofPropertyComputer] = None,
        verbose: bool = True) -> Sequence[float]:
    """Extract the property. Computes it and saves if not present.
    Computes only missing folds and saves them individually as soon as ready."""
    registry = validation_registry_from_hof_path(hof_path=hof_path)
    n_folds = n_folds_from_files(hof_dir=hof_path)
    if registry.has_property(name=prop_name):
        res = registry.get_property(name=prop_name)
        if is_sequence_not_string(res):
            if len(res) == n_folds:
                for i in range(n_folds):
                    if res[i] is None:
                        if verbose:
                            print("Property " + prop_name + " found in registry in directory " + hof_path + " but...")
                            print("One of the elements is missing, computing it and restarting the function.")
                        if computer is not None:
                            res[i] = computer.compute_fold(hof_path=hof_path, fold=i)
                            registry.set_property(name=prop_name, value=res)
                            return __smart_extract_folds(prop_name=prop_name, hof_path=hof_path, computer=computer,
                                                         verbose=verbose)
                        else:
                            if verbose:
                                print("Computer not found for the property.")
                            raise KeyError()
                return res
            else:
                if verbose:
                    print("Property " + prop_name + " found in registry in directory " + hof_path + " but...")
                    print("The number of elements is different from the number of folds.")
                    print("Setting all elements to None and restarting the function.")
                res = [None]*n_folds
        else:
            if verbose:
                print("The value is not a non-string Sequence.")
                print("Setting all elements to None and restarting the function.")
            res = [None] * n_folds
    else:
        if verbose:
            print("Property " + prop_name + " not found in registry in directory " + hof_path)
            print("Setting all elements to None and restarting the function.")
        res = [None] * n_folds
    registry.set_property(name=prop_name, value=res)
    return __smart_extract_folds(prop_name=prop_name, hof_path=hof_path, computer=computer, verbose=verbose)


def smart_extract(
        prop_name: str, hof_path: str, computer: Optional[HofPropertyComputer] = None, verbose: bool = True) -> Any:
    """Extract the property. Computes it and saves if not present."""
    if computer is not None and computer.has_folds():
        return __smart_extract_folds(prop_name=prop_name, hof_path=hof_path, computer=computer)
    else:
        registry = validation_registry_from_hof_path(hof_path=hof_path)
        if registry.has_property(name=prop_name):
            return registry.get_property(name=prop_name)
        else:
            if verbose:
                print("Property " + prop_name + " not found in registry in directory " + hof_path)
            if computer is not None:
                if verbose:
                    print("Computing and saving the property.")
                res = computer.compute(hof_path=hof_path)
                registry.set_property(name=prop_name, value=res)
                return res
            else:
                if verbose:
                    print("Computer not found for the property.")
                raise KeyError()
