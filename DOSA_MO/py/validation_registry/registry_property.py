from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from plots.saved_hof import validation_registry_from_hof_path
from saved_solutions.solutions_from_files import n_folds_from_files
from util.backed_dict.backed_dict_property import BackedDictProperty
from util.utils import IllegalStateError, is_sequence_not_string
from validation_registry.hof_property_computer import HofPropertyComputer, HofPropertyComputerWithFolds
from validation_registry.validation_registry import ValidationRegistry
from validation_registry.allowed_property_names import ALLOWED_PROPERTY_NAMES


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


class RegistryProperty:
    __inner: BackedDictProperty
    __folds_property_name: Optional[str]
    __computer: Optional[HofPropertyComputer]
    __folds_computer: Optional[HofPropertyComputerWithFolds]

    def __init__(self, property_name: str, nick_for_humans: str, folds_property_name: Optional[str] = None,
                 computer: Optional[HofPropertyComputer] = None,
                 folds_computer: Optional[HofPropertyComputerWithFolds] = None):
        if property_name in ALLOWED_PROPERTY_NAMES:
            self.__inner = BackedDictProperty(
                property_name=property_name, name_for_humans=property_name, nick_for_humans=nick_for_humans)
            self.__folds_property_name = folds_property_name
        else:
            message = "Passed property name is not allowed: " + str(property_name)
            raise ValueError(message)
        self.__computer = computer
        self.__folds_computer = folds_computer

    def property_name(self) -> str:
        return self.__inner.property_name()

    def name(self) -> str:
        """Name for humans."""
        return self.__inner.name()

    def nick(self) -> str:
        return self.__inner.nick()

    def __str__(self) -> str:
        return self.name()

    def extract(self, registry: ValidationRegistry) -> Any:
        return registry.get_property(name=self.property_name())

    def extract_or_none(self, registry: ValidationRegistry) -> Any:
        return registry.get_property_or_none(name=self.property_name())

    def extract_folds_property(self, registry: ValidationRegistry) -> Any:
        if self.__folds_property_name is not None:
            return registry.get_property(name=self.__folds_property_name)
        else:
            raise IllegalStateError()

    def has_folds_property_name(self) -> bool:
        return self.__folds_property_name is not None

    def folds_property_name(self) -> str:
        """Name for humans."""
        if self.has_folds_property_name():
            return self.__folds_property_name
        else:
            raise IllegalStateError()

    def has_property(self, registry: ValidationRegistry) -> bool:
        return registry.has_property(name=self.property_name())

    def has_folds_property(self, registry: ValidationRegistry) -> bool:
        if self.has_folds_property_name():
            return registry.has_property(name=self.__folds_property_name)
        else:
            return False

    def can_compute(self) -> bool:
        return self.__computer is not None

    def can_compute_for_folds(self) -> bool:
        return self.__folds_computer is not None

    def compute(self, hof_path: str) -> Any:
        """Forces the computation: registry entry is not used even if present.
        Still, it may use other entries from which the property derives."""
        return self.__computer.compute(hof_path=hof_path)

    def compute_for_folds(self, hof_path: str) -> Any:
        """Forces the computation: registry entry is not used even if present.
        Still, it may use other entries from which the property derives."""
        return self.__folds_computer.compute(hof_path=hof_path)

    def smart_extract(self, hof_path: str, verbose: bool = True) -> Any:
        """Extracts the value. Computes and saves if necessary. Computes and saves incrementally if possible."""
        prop_name = self.property_name()
        return smart_extract(prop_name=prop_name, hof_path=hof_path, computer=self.__computer, verbose=verbose)

    def smart_extract_folds(self, hof_path: str, verbose: bool = True) -> Any:
        prop_name = self.folds_property_name()
        return smart_extract(prop_name=prop_name, hof_path=hof_path, computer=self.__folds_computer, verbose=verbose)
