from abc import ABC
from typing import Any

from util.backed_dict.backed_dict import BackedDict, MemoryBackedDict, FileBackedDict
from validation_registry.allowed_property_names import ALLOWED_PROPERTY_NAMES, RENAMES


class ValidationRegistry(ABC):
    __backend: BackedDict

    def __init__(self, back_end: BackedDict):
        self.__backend = back_end

    def set_property(self, name: str, value):
        self.__backend.set(property_name=name, property_value=value)

    def get_property(self, name: str) -> Any:
        return self.__backend.get(property_name=name)

    def has_property(self, name: str) -> bool:
        return self.__backend.has(property_name=name)

    def get_property_or_none(self, name: str) -> Any:
        if self.has_property(name=name):
            return self.get_property(name=name)
        else:
            return None

    def clean(self):
        self.__backend.clean()


class FileValidationRegistry(ValidationRegistry):

    def __init__(self, file_path: str):
        ValidationRegistry.__init__(self, back_end=FileBackedDict(
            file_path=file_path, allowed_property_names=ALLOWED_PROPERTY_NAMES, to_rename=RENAMES))


class MemoryValidationRegistry(ValidationRegistry):

    def __init__(self):
        ValidationRegistry.__init__(self, back_end=MemoryBackedDict(allowed_property_names=ALLOWED_PROPERTY_NAMES))
