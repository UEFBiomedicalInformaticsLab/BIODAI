from descriptor.described_generator import NestedDescribedGenerator, \
    DescribedGeneratorRegistry, DescribedGeneratorForAClass
from descriptor.descriptor import Descriptor, Described
from univariate_feature_selection.fdr_many_feature_selector import FdrManyFeatureSelector
from univariate_feature_selection.feature_selector_multi_target import FeatureSelectorMOUnion, DummySelectorMO, \
    DUMMY_SELECTOR
from univariate_feature_selection.many_feature_selector import CompositeManyFeatureSelector, \
    ManyFeatureSelectorAnovaCategorical, ManyFeatureSelectorCox, ManyFeatureSelector, ManyFeatureSelectorPipeline, \
    DUMMY_SELECTOR_MANY, DummyManyFeatureSelector, ManyFeatureSelectorWithWorkers
from univariate_feature_selection.single_feature_selector import SingleFeatureSelector, MAFFeatureSelector, \
    HWESingleFeatureSelector
from univariate_feature_selection.univariate_feature_selector_descriptor import CompositeManyFeatureSelectorDescriptor, \
    AnovaCategoricalDescriptor, FeatureSelectorCoxDescriptor, FeatureSelectorMOUnionDescriptor, \
    ManyFeatureSelectorPipelineDescriptor, DummyManyFeatureSelectorDescriptor, FdrManyFeatureSelectorDescriptor, \
    DummySelectorMODescriptor, FdrManyFeatureSelectorClassDescriptor, ManyFeatureSelectorFromSingleDescriptor, \
    MAFSingleFeatureSelectorDescriptor, HWESingleFeatureSelectorDescriptor
from univariate_property_computer.univariate_property_computer_descriptor import LogUnivariatePvalComputerDescriptor, \
    AnovaUnivariatePvalComputerDescriptor
from univariate_property_computer.univariate_pval_computer import UnivariatePvalComputer, LogUnivariatePvalComputer, \
    AnovaUnivariatePvalComputer


class DummySelectorManyGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=DummyManyFeatureSelectorDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> DummyManyFeatureSelector:
        return DUMMY_SELECTOR_MANY


class CompositeManyFeatureSelectorGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=CompositeManyFeatureSelectorDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> CompositeManyFeatureSelector:
        assert isinstance(descriptor, CompositeManyFeatureSelectorDescriptor)
        categorical = self.generate_by_registry(descriptor.categorical_selector())
        survival = self.generate_by_registry(descriptor.survival_selector())
        assert isinstance(categorical, ManyFeatureSelector)
        assert isinstance(survival, ManyFeatureSelector)
        return CompositeManyFeatureSelector(categorical_selector=categorical, survival_selector=survival)


class AnovaCategoricalGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=AnovaCategoricalDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> ManyFeatureSelectorAnovaCategorical:
        if isinstance(descriptor, AnovaCategoricalDescriptor):
            return ManyFeatureSelectorAnovaCategorical(p_val=descriptor.p_val())
        else:
            raise TypeError()


class FeatureSelectorCoxGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=FeatureSelectorCoxDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> ManyFeatureSelectorCox:
        if isinstance(descriptor, FeatureSelectorCoxDescriptor):
            return ManyFeatureSelectorCox(p_val=descriptor.p_val())
        else:
            raise TypeError()


class HWESingleFeatureSelectorGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=HWESingleFeatureSelectorDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> HWESingleFeatureSelector:
        assert isinstance(descriptor, HWESingleFeatureSelectorDescriptor)
        return HWESingleFeatureSelector(control_class=descriptor.control_class(), p_val=descriptor.p_val())


class MAFSingleFeatureSelectorGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=MAFSingleFeatureSelectorDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> MAFFeatureSelector:
        assert isinstance(descriptor, MAFSingleFeatureSelectorDescriptor)
        return MAFFeatureSelector(threshold=descriptor.threshold())


class FdrManyFeatureSelectorGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=FdrManyFeatureSelectorDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> Described:
        assert isinstance(descriptor, FdrManyFeatureSelectorDescriptor)
        computer = self.generate_by_registry(descriptor=descriptor.computer())
        assert isinstance(computer, UnivariatePvalComputer)
        return FdrManyFeatureSelector(computer=computer, fdr_threshold=descriptor.fdr_threshold())


class FdrManyFeatureSelectorClassGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=FdrManyFeatureSelectorClassDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> Described:
        assert isinstance(descriptor, FdrManyFeatureSelectorClassDescriptor)
        computer = self.generate_by_registry(descriptor=descriptor.computer())
        assert isinstance(computer, UnivariatePvalComputer)
        return FdrManyFeatureSelector(computer=computer, fdr_threshold=descriptor.fdr_threshold())


class ManyFeatureSelectorPipelineGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=ManyFeatureSelectorPipelineDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> Described:
        if isinstance(descriptor, ManyFeatureSelectorPipelineDescriptor):
            selectors = []
            for d in descriptor.selectors():
                selector = self.generate_by_registry(descriptor=d)
                assert isinstance(selector, ManyFeatureSelector)
                selectors.append(selector)
            return ManyFeatureSelectorPipeline(selectors=selectors)
        else:
            raise TypeError()


class FeatureSelectorMOUnionGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=FeatureSelectorMOUnionDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> FeatureSelectorMOUnion:
        assert isinstance(descriptor, FeatureSelectorMOUnionDescriptor)
        feature_selector_so = self.generate_by_registry(descriptor=descriptor.feature_selector_so())
        assert isinstance(feature_selector_so, ManyFeatureSelector)
        return FeatureSelectorMOUnion(feature_selector_so=feature_selector_so)


class DummySelectorMOGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=DummySelectorMODescriptor)

    def inner_generate(self, descriptor: Descriptor) -> DummySelectorMO:
        return DUMMY_SELECTOR


class LogUnivariatePvalComputerGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=LogUnivariatePvalComputerDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> LogUnivariatePvalComputer:
        return LogUnivariatePvalComputer()


class AnovaUnivariatePvalComputerGenerator(DescribedGeneratorForAClass):

    def __init__(self):
        DescribedGeneratorForAClass.__init__(self=self, descriptor_class=AnovaUnivariatePvalComputerDescriptor)

    def inner_generate(self, descriptor: Descriptor) -> AnovaUnivariatePvalComputer:
        return AnovaUnivariatePvalComputer()


class ManyFeatureSelectorFromSingleGenerator(NestedDescribedGenerator):

    def __init__(self, registry: DescribedGeneratorRegistry):
        NestedDescribedGenerator.__init__(
            self=self, descriptor_class=ManyFeatureSelectorFromSingleDescriptor, registry=registry)

    def inner_generate(self, descriptor: Descriptor) -> Described:
        if isinstance(descriptor, ManyFeatureSelectorFromSingleDescriptor):
            selector = self.generate_by_registry(descriptor.single_feature_selector())
            assert isinstance(selector, SingleFeatureSelector)
            return ManyFeatureSelectorWithWorkers(single_fs=selector)
        else:
            raise TypeError()


DEFAULT_UNIVARIATE_FS_GENERATOR = DescribedGeneratorRegistry()
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=DummySelectorManyGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=AnovaCategoricalGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=FeatureSelectorCoxGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=FdrManyFeatureSelectorClassGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=FdrManyFeatureSelectorGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=ManyFeatureSelectorPipelineGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=CompositeManyFeatureSelectorGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=DummySelectorMOGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=FeatureSelectorMOUnionGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=LogUnivariatePvalComputerGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=AnovaUnivariatePvalComputerGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(
    generator=ManyFeatureSelectorFromSingleGenerator(registry=DEFAULT_UNIVARIATE_FS_GENERATOR))
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=MAFSingleFeatureSelectorGenerator())
DEFAULT_UNIVARIATE_FS_GENERATOR.register_generator(generator=HWESingleFeatureSelectorGenerator())
