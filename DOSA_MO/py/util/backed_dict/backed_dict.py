import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any, Optional


class BackedDict(ABC):

    @abstractmethod
    def set(self, property_name: str, property_value: Any):
        raise NotImplementedError()

    @abstractmethod
    def get(self, property_name: str) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def has(self, property_name: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def clean(self):
        raise NotImplementedError()


class CheckedBackedDict(BackedDict):
    __allowed_property_names: Optional[set[str]]
    __best_effort: bool

    def __init__(self, allowed_property_names: Optional[set[str]] = None, best_effort: bool = True):
        self.__allowed_property_names = allowed_property_names
        self.__best_effort = best_effort

    def set(self, property_name: str, property_value: Any):
        """Saves immediately upon set."""
        if self.is_allowed(property_name):
            self._inner_set(property_name=property_name, property_value=property_value)
        else:
            message = "Passed property name is not allowed: " + str(property_name)
            if self.__best_effort:
                print(message)
            else:
                raise ValueError(message)

    @abstractmethod
    def _inner_set(self, property_name: str, property_value: Any):
        raise NotImplementedError()

    def is_allowed(self, property_name: str) -> bool:
        return self.__allowed_property_names is None or property_name in self.__allowed_property_names

    def _best_effort(self) -> bool:
        return self.__best_effort


class FileBackedDict(CheckedBackedDict):
    """We assume there is only one instance for each file and no one else writes on the file."""
    __file_path: str
    __registry: dict
    __to_rename: Sequence[tuple[str, str]]  # Rename from element 0 to element 1
    __to_reset: Sequence[str]

    def __init__(self, file_path: str, allowed_property_names: Optional[set[str]] = None, best_effort: bool = True,
                 to_rename: Sequence[tuple[str, str]] = (), to_reset: Sequence[str] = ()):
        """If best_effort, when there are errors it just does not write the updated dictionary."""
        CheckedBackedDict.__init__(self, allowed_property_names=allowed_property_names, best_effort=best_effort)
        self.__file_path = file_path
        self.__to_rename = to_rename
        self.__to_reset = to_reset
        self.__load()

    def _inner_set(self, property_name: str, property_value: Any):
        """Reloads before saving. Saves immediately upon set."""
        self.__load()
        self.__registry[property_name] = property_value
        self.__save()

    def get(self, property_name: str) -> Any:
        self.__load()
        return self.__registry[property_name]

    def has(self, property_name: str) -> bool:
        self.__load()
        return property_name in self.__registry

    def __save(self):
        try:
            os.makedirs(os.path.dirname(self.__file_path), exist_ok=True)
            with open(self.__file_path, 'w') as f:
                json.dump(self.__registry, f)
        except (OSError, TypeError, ValueError) as e:
            if self.__best_effort:
                print("Error while saving to file " + str(self.__file_path) + ": " + str(e))
            else:
                raise e

    def __load(self):
        """Does not load not allowed keys."""
        changed = False
        try:
            with open(self.__file_path) as f:
                self.__registry = json.load(f)
                for unwanted in self.__to_reset:
                    if unwanted in self.__registry:
                        del self.__registry[unwanted]
                        changed = True
                for rule in self.__to_rename:
                    if rule[0] in self.__registry:
                        self.__registry[rule[1]] = self.__registry[rule[0]]
                        del self.__registry[rule[0]]
                        changed = True
                to_remove = []
                for k in self.__registry:
                    if not self.is_allowed(property_name=k):
                        to_remove.append(k)
                for k in to_remove:
                    del self.__registry[k]
                    changed = True
        except OSError:
            self.__registry = {}
        except JSONDecodeError as e:
            if self._best_effort():
                print("Error while decoding JSON of file " + str(self.__file_path) + ": " + str(e))
                self.__registry = {}
            else:
                raise e
        if changed:
            self.__save()

    def clean(self):
        self.__registry = {}
        self.__save()


class MemoryBackedDict(CheckedBackedDict):
    __registry: dict

    def __init__(self, allowed_property_names: Optional[set[str]] = None, best_effort: bool = True):
        CheckedBackedDict.__init__(self, allowed_property_names=allowed_property_names, best_effort=best_effort)
        self.__registry = {}

    def _inner_set(self, property_name: str, property_value):
        self.__registry[property_name] = property_value

    def get(self, property_name: str) -> Any:
        return self.__registry[property_name]

    def has(self, property_name: str) -> bool:
        return property_name in self.__registry

    def clean(self):
        self.__registry = {}
