from __future__ import annotations
from abc import abstractmethod, ABC
from copy import deepcopy

from descriptor.descriptor import Descriptor, Described
from util.str_utils import iterable_to_string


class DescribedGenerator(ABC):

    @abstractmethod
    def generate(self, descriptor: Descriptor) -> Described:
        raise NotImplementedError()


class DescribedGeneratorForAClass(DescribedGenerator, ABC):
    __descriptor_class: type

    def __init__(self, descriptor_class: type):
        if issubclass(descriptor_class, Descriptor):
            self.__descriptor_class = descriptor_class
        else:
            raise TypeError("Unexpected class: not a subclass of Descriptor. Class: " + str(descriptor_class))

    def descriptor_class(self) -> type:
        return self.__descriptor_class

    def generate(self, descriptor: Descriptor) -> Described:
        if isinstance(descriptor, self.descriptor_class()):
            assert isinstance(descriptor, Descriptor)
            return self.inner_generate(descriptor=descriptor)
        else:
            raise TypeError("Unexpected descriptor type.")

    @abstractmethod
    def inner_generate(self, descriptor: Descriptor) -> Described:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Described generator for descriptors of type " + str(self.descriptor_class())


class DescribedGeneratorRegistry(DescribedGenerator):
    __registry: dict[type,DescribedGeneratorForAClass]

    def __init__(self):
        self.__registry = dict()

    def register_generator(self, generator: DescribedGeneratorForAClass):
        self.__registry[generator.descriptor_class()] = generator

    def generate(self, descriptor: Descriptor) -> Described:
        descriptor_type = type(descriptor)
        generator = self.__registry.get(descriptor_type)
        if generator is None:
            raise TypeError("No registered generator for descriptors of type " + str(descriptor_type))
        else:
            return generator.generate(descriptor=descriptor)

    def __str__(self) -> str:
        return ("Described generator registry with registered generators for types " +
                iterable_to_string(li=self.__registry.keys()))

    def __copy__(self) -> DescribedGeneratorRegistry:
        res = DescribedGeneratorRegistry()
        for g in self.__registry.values():
            res.register_generator(generator=g)
        return res

    def __deepcopy__(self, memo) -> DescribedGeneratorRegistry:
        res = DescribedGeneratorRegistry()
        for g in self.__registry.values():
            res.register_generator(generator=deepcopy(g))
        return res


class NestedDescribedGenerator(DescribedGeneratorForAClass, ABC):
    """Used to generate a described that is composed of other described objects."""
    __registry: DescribedGeneratorRegistry

    def __init__(self, descriptor_class: type, registry: DescribedGeneratorRegistry):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=descriptor_class)
        self.__registry = registry

    def generate_by_registry(self, descriptor: Descriptor) -> Described:
        return self.__registry.generate(descriptor=descriptor)
