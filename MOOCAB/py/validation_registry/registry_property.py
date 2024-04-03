from __future__ import annotations

from typing import Any, Optional

from util.backed_dict.backed_dict_property import BackedDictProperty
from util.utils import IllegalStateError, is_sequence_not_string
from validation_registry.hof_property_computer import HofPropertyComputer, HofPropertyComputerWithFolds
from validation_registry.hof_property_computer_folds_mean import HofPropertyComputerFoldsMeanFromRegistry
from validation_registry.smart_extract import smart_extract
from validation_registry.validation_registry import ValidationRegistry
from validation_registry.allowed_property_names import ALLOWED_PROPERTY_NAMES


class RegistryProperty:
    """A property that can either be computed for single folds or as an aggregation (e.g. a mean) of all the
    folds, or both."""
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

    def has_complete_folds_property(self, registry: ValidationRegistry) -> bool:
        """True only if all folds are computed. Does not check if the number of folds of the property is consistent
        with the result tables in the saved hof directory."""
        if self.has_folds_property_name():
            prop_name = self.__folds_property_name
            if registry.has_property(name=prop_name):
                prop = registry.get_property(name=prop_name)
                if is_sequence_not_string(prop):
                    for f in prop:
                        if f is None:
                            return False
                    return True
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


class RegistryPropertyWithMean(RegistryProperty):
    """The property for the aggregated folds is the mean of the property for the single folds."""

    def __init__(self, property_name: str, nick_for_humans: str, folds_property_name: Optional[str] = None,
                 folds_computer: Optional[HofPropertyComputerWithFolds] = None):
        RegistryProperty.__init__(self=self, property_name=property_name, nick_for_humans=nick_for_humans,
                                  folds_property_name=folds_property_name,
                                  computer=HofPropertyComputerFoldsMeanFromRegistry(
                                      folds_property_name=folds_property_name,
                                      folds_property_computer=folds_computer),
                                  folds_computer=folds_computer)