from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Any

from consts import DEFAULT_P_VAL
from util.str_utils import fdr_str, str_paste, proportion_str, iterable_to_string
from univariate_property_computer.univariate_property_computer_descriptor import UnivariatePvalComputerDescriptor, \
    LOG_UNIVARIATE_NICK
from descriptor.descriptor import Descriptor
from util.utils import IllegalStateError

ANOVA_NICK = "anova"
ANOVA_NAME = "ANOVA"
DEFAULT_CATEGORICAL_FS_NICK = ANOVA_NICK
DEFAULT_FDR_THRESHOLD = 0.05
FDR_STR = "FDR"
DEFAULT_MISSING_THRESHOLD = 0.05
DEFAULT_MINOR_FREQUENCY_THRESHOLD = 0.01
DEFAULT_MAF_THRESHOLD = 0.01
DEFAULT_HWE_PVAL = 0.000001

LOGISTIC_FDR_SELECTOR_NICK = LOG_UNIVARIATE_NICK + FDR_STR
"""Does not include the threshold value."""


class SingleFeatureSelectorDescriptor(Descriptor, ABC):

    @abstractmethod
    def uses_covariates(self) -> bool:
        raise NotImplementedError()

    def algorithm_name(self) -> str:
        return self.algorithm_nick()

    @abstractmethod
    def algorithm_nick(self) -> str:
        raise NotImplementedError()


class DummySingleFeatureSelectorDescriptor(SingleFeatureSelectorDescriptor):

    def __str__(self) -> str:
        return "Dummy single feature selector"

    def nick(self) -> str:
        return self.algorithm_nick()

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "dummy"


class MissingSingleFeatureSelectorDescriptor(SingleFeatureSelectorDescriptor):
    __threshold: float

    def __init__(self, threshold: float = DEFAULT_MISSING_THRESHOLD):
        self.__threshold = threshold

    def threshold(self) -> float:
        return self.__threshold

    def __str__(self) -> str:
        return "Missingness feature selector with threshold " + proportion_str(proportion=self.__threshold)

    def nick(self) -> str:
        return self.algorithm_nick() + proportion_str(proportion=self.__threshold)

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "miss"


class MinorFrequencySingleFeatureSelectorDescriptor(SingleFeatureSelectorDescriptor):
    __threshold: float

    def __init__(self, threshold: float = DEFAULT_MINOR_FREQUENCY_THRESHOLD):
        self.__threshold = threshold

    def threshold(self) -> float:
        return self.__threshold

    def __str__(self) -> str:
        return "Minor frequency feature selector with threshold " + proportion_str(proportion=self.__threshold)

    def nick(self) -> str:
        return self.algorithm_nick() + proportion_str(proportion=self.__threshold)

    def algorithm_nick(self) -> str:
        return "MF"

    def uses_covariates(self) -> bool:
        return False


class MAFSingleFeatureSelectorDescriptor(SingleFeatureSelectorDescriptor):
    __threshold: float

    def __init__(self, threshold: float = DEFAULT_MAF_THRESHOLD):
        self.__threshold = threshold

    def threshold(self) -> float:
        return self.__threshold

    def __str__(self) -> str:
        return "minor allele frequency feature selector with threshold " + proportion_str(proportion=self.__threshold)

    def nick(self) -> str:
        return self.algorithm_nick() + proportion_str(proportion=self.__threshold)

    def algorithm_nick(self) -> str:
        return "MAF"

    def uses_covariates(self) -> bool:
        return False



class WithPvalDescriptor(Descriptor, ABC):
    __p_val: float

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        self.__p_val = p_val

    def p_val(self) -> float:
        return self.__p_val

    def p_val_nick(self) -> str:
        return proportion_str(proportion=self.__p_val)


class SingleFeatureSelectorWithPvalDescriptor(SingleFeatureSelectorDescriptor, WithPvalDescriptor, ABC):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        WithPvalDescriptor.__init__(self=self, p_val=p_val)

    def nick(self) -> str:
        return self.algorithm_nick() + self.p_val_nick()

    def name(self) -> str:
        return self.algorithm_name() + " " + self.p_val_nick()


class SingleFeatureSelectorAnovaCategoricalDescriptor(SingleFeatureSelectorWithPvalDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "Categorical single feature selector any na and anova with p-value " + self.p_val_nick()

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "anova"


class SingleFeatureSelectorAnovaSurvivalDescriptor(SingleFeatureSelectorWithPvalDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "Single feature selector any na and anova on survival events with p-value " + self.p_val_nick()

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "anova"


class SingleFeatureSelectorCoxDescriptor(SingleFeatureSelectorWithPvalDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        SingleFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "Single feature selector Cox with p-value " + self.p_val_nick()

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "cox"


class HWESingleFeatureSelectorDescriptor(SingleFeatureSelectorWithPvalDescriptor):
    __control_class: Optional[Any]

    def __init__(self, control_class: Optional[Any] = None, p_val: float = DEFAULT_HWE_PVAL):
        SingleFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)
        self.__control_class = control_class

    def __str__(self) -> str:
        res = "Hardy-Weinberg Equilibrium feature selector with p-value " + self.p_val_nick()
        if self.__control_class is not None:
            res += " and control class " + str(self.__control_class)
        return res

    def control_class(self) -> Any:
        return self.__control_class

    def uses_covariates(self) -> bool:
        return False

    def algorithm_nick(self) -> str:
        return "HWE"


class CompositeSingleFeatureSelectorDescriptor(SingleFeatureSelectorDescriptor):
    __categorical_selector: SingleFeatureSelectorDescriptor
    __survival_selector: SingleFeatureSelectorDescriptor

    def __init__(
            self,
            categorical_selector: SingleFeatureSelectorDescriptor,
            survival_selector: SingleFeatureSelectorDescriptor):
        self.__categorical_selector = categorical_selector
        self.__survival_selector = survival_selector

    def categorical_selector(self) -> SingleFeatureSelectorDescriptor:
        return self.__categorical_selector

    def survival_selector(self) -> SingleFeatureSelectorDescriptor:
        return self.__survival_selector

    def __str__(self) -> str:
        res = "composite single feature selector with\n"
        res += "categorical: " + str(self.__categorical_selector) + "\n"
        res += "survival: " + str(self.__survival_selector) + "\n"
        return res

    def name(self) -> str:
        return "(" + self.__categorical_selector.name() + ", " + self.__survival_selector.name() + ")"

    def nick(self) -> str:
        return "(" + self.__categorical_selector.nick() + "," + self.__survival_selector.nick() + ")"

    def algorithm_name(self) -> str:
        return "(" + self.__categorical_selector.algorithm_name() + ", " + self.__survival_selector.algorithm_name() + ")"

    def algorithm_nick(self) -> str:
        return "(" + self.__categorical_selector.algorithm_nick() + "," + self.__survival_selector.algorithm_nick() + ")"

    def uses_covariates(self) -> bool:
        return self.__categorical_selector.uses_covariates() or self.__survival_selector.uses_covariates()


class ManyFeatureSelectorDescriptor(Descriptor, ABC):

    @abstractmethod
    def uses_covariates(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_fdr_threshold(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def fdr_threshold(self) -> float:
        raise NotImplementedError()

    def fdr_str(self) -> str:
        if self.has_fdr_threshold():
            return fdr_str(fdr_threshold=self.fdr_threshold())
        else:
            return ""

    def algorithm_name(self) -> str:
        return self.algorithm_nick()

    @abstractmethod
    def algorithm_nick(self) -> str:
        """Only algorithm nick, without any parameters."""
        raise NotImplementedError()


class ManyFeatureSelectorClassDescriptor(ManyFeatureSelectorDescriptor, ABC):
    pass


class ManyFeatureSelectorSurvDescriptor(ManyFeatureSelectorDescriptor, ABC):
    pass


class CompositeManyFeatureSelectorDescriptor(ManyFeatureSelectorDescriptor):
    __categorical_selector: ManyFeatureSelectorClassDescriptor
    __survival_selector: ManyFeatureSelectorSurvDescriptor

    def __init__(self,
                 categorical_selector: ManyFeatureSelectorClassDescriptor,
                 survival_selector: ManyFeatureSelectorSurvDescriptor):
        self.__categorical_selector = categorical_selector
        self.__survival_selector = survival_selector

    def categorical_selector(self) -> ManyFeatureSelectorClassDescriptor:
        return self.__categorical_selector

    def survival_selector(self) -> ManyFeatureSelectorSurvDescriptor:
        return self.__survival_selector

    def __str__(self) -> str:
        res = "composite many feature selector with\n"
        res += "categorical: " + str(self.__categorical_selector) + "\n"
        res += "survival: " + str(self.__survival_selector) + "\n"
        return res

    def name(self) -> str:
        return "(" + self.__categorical_selector.name() + ", " + self.__survival_selector.name() + ")"

    def nick(self) -> str:
        return "(" + self.__categorical_selector.nick() + "," + self.__survival_selector.nick() + ")"

    def uses_covariates(self) -> bool:
        return self.__categorical_selector.uses_covariates() or self.__survival_selector.uses_covariates()

    def has_fdr_threshold(self) -> bool:
        """Returns false. Even if both categorical and survival descriptors have fdr, they could be different."""
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def algorithm_name(self) -> str:
        return ("(" + self.__categorical_selector.algorithm_name() +
                ", " + self.__survival_selector.algorithm_name() + ")")

    def algorithm_nick(self) -> str:
        return ("(" + self.__categorical_selector.algorithm_nick() +
                "," + self.__survival_selector.algorithm_nick() + ")")


class ManyFeatureSelectorWithPvalDescriptor(ManyFeatureSelectorDescriptor, WithPvalDescriptor, ABC):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        WithPvalDescriptor.__init__(self=self, p_val=p_val)

    def has_fdr_threshold(self) -> bool:
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def nick(self) -> str:
        return self.algorithm_nick() + self.p_val_nick()

    def name(self) -> str:
        return self.algorithm_name() + " " + self.p_val_nick()


class AnovaCategoricalDescriptor(ManyFeatureSelectorWithPvalDescriptor, ManyFeatureSelectorClassDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "categorical feature selector any na and anova"

    def uses_covariates(self) -> bool:
        return False

    def has_fdr_threshold(self) -> bool:
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def algorithm_name(self) -> str:
        return ANOVA_NAME

    def algorithm_nick(self) -> str:
        return ANOVA_NICK


class AnovaSurvivalDescriptor(ManyFeatureSelectorWithPvalDescriptor, ManyFeatureSelectorSurvDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "feature selector any na and anova on survival events"

    def uses_covariates(self) -> bool:
        return False

    def has_fdr_threshold(self) -> bool:
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def algorithm_name(self) -> str:
        return ANOVA_NAME

    def algorithm_nick(self) -> str:
        return ANOVA_NICK


class FeatureSelectorCoxDescriptor(ManyFeatureSelectorWithPvalDescriptor, ManyFeatureSelectorSurvDescriptor):

    def __init__(self, p_val: float = DEFAULT_P_VAL):
        ManyFeatureSelectorWithPvalDescriptor.__init__(self=self, p_val=p_val)

    def __str__(self) -> str:
        return "feature selector Cox"

    def uses_covariates(self) -> bool:
        return False

    def algorithm_name(self) -> str:
        return "Cox"

    def algorithm_nick(self) -> str:
        return "cox"


class FeatureSelectorMODescriptor(Descriptor, ABC):
    pass

class FeatureSelectorMOUnionDescriptor(FeatureSelectorMODescriptor):
    __feature_selector_so: ManyFeatureSelectorDescriptor

    def __init__(self, feature_selector_so: ManyFeatureSelectorDescriptor):
        self.__feature_selector_so = feature_selector_so

    def feature_selector_so(self) -> ManyFeatureSelectorDescriptor:
        return self.__feature_selector_so

    def __str__(self) -> str:
        return "Multi-objective feature selector with inner " + str(self.__feature_selector_so)

    def name(self) -> str:
        return "MO " + self.__feature_selector_so.name()

    def nick(self) -> str:
        return self.__feature_selector_so.nick()


class DummyManyFeatureSelectorDescriptor(ManyFeatureSelectorDescriptor):

    def __str__(self) -> str:
        return "Dummy many feature selector"

    def name(self) -> str:
        return self.algorithm_name()

    def nick(self) -> str:
        return self.algorithm_nick()

    def uses_covariates(self) -> bool:
        return False

    def has_fdr_threshold(self) -> bool:
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def algorithm_name(self) -> str:
        return "dummy FS"

    def algorithm_nick(self) -> str:
        return "dummy"


class DummySelectorMODescriptor(FeatureSelectorMODescriptor):
    """Accepts every feature."""

    def __str__(self) -> str:
        return "Dummy multi-objective feature selector"

    def name(self) -> str:
        return "MO dummy FS"

    def nick(self) -> str:
        return "dummy"


class FdrManyFeatureSelectorDescriptor(ManyFeatureSelectorDescriptor):
    __computer: UnivariatePvalComputerDescriptor
    __fdr_threshold: float

    def __init__(self,
                 computer: UnivariatePvalComputerDescriptor,
                 fdr_threshold: float = DEFAULT_FDR_THRESHOLD):
        self.__computer = computer
        self.__fdr_threshold = fdr_threshold

    def __str__(self) -> str:
        return ("FDR many feature selector with p-val computer " +
                str(self.__computer) + " and FDR threshold " + self.fdr_str())

    def computer(self) -> UnivariatePvalComputerDescriptor:
        return self.__computer

    def algorithm_nick(self) -> str:
        return self.computer().nick() + FDR_STR

    def nick(self) -> str:
        return self.algorithm_nick() + self.fdr_str()

    def name(self) -> str:
        return self.__computer.name() + " " + FDR_STR + " " + self.fdr_str()

    def uses_covariates(self) -> bool:
        return self.__computer.uses_covariates()

    def has_fdr_threshold(self) -> bool:
        return True

    def fdr_threshold(self) -> float:
        return self.__fdr_threshold

    def algorithm_name(self) -> str:
        return self.__computer.name() + " " + FDR_STR


class FdrManyFeatureSelectorClassDescriptor(FdrManyFeatureSelectorDescriptor, ManyFeatureSelectorClassDescriptor):

    def __init__(self,
                 computer: UnivariatePvalComputerDescriptor,
                 fdr_threshold: float = DEFAULT_FDR_THRESHOLD):
        FdrManyFeatureSelectorDescriptor.__init__(self=self, computer=computer, fdr_threshold=fdr_threshold)


class ManyFeatureSelectorFromSingleDescriptor(ManyFeatureSelectorDescriptor):
    __single_fs: SingleFeatureSelectorDescriptor

    def __init__(self, single_fs: SingleFeatureSelectorDescriptor):
        self.__single_fs = single_fs

    def __str__(self) -> str:
        return "Many feature selector using " + str(self.__single_fs)

    def nick(self) -> str:
        return self.__single_fs.nick()

    def name(self) -> str:
        return self.__single_fs.name()

    def uses_covariates(self) -> bool:
        return self.__single_fs.uses_covariates()

    def has_fdr_threshold(self) -> bool:
        """Returns false. Fdr does not make sense with just one feature at a time."""
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def algorithm_name(self) -> str:
        return self.__single_fs.algorithm_name()

    def algorithm_nick(self) -> str:
        return self.__single_fs.algorithm_nick()

    def single_feature_selector(self) -> SingleFeatureSelectorDescriptor:
        return self.__single_fs


class ManyFeatureSelectorPipelineDescriptor(ManyFeatureSelectorDescriptor):
    __selectors: Sequence[ManyFeatureSelectorDescriptor]

    def __init__(self, selectors: Sequence[ManyFeatureSelectorDescriptor]):
        self.__selectors = list(selectors)

    def selectors(self) -> Sequence[ManyFeatureSelectorDescriptor]:
        return self.__selectors

    def nick(self) -> str:
        return str_paste(parts=[s.nick() for s in self.__selectors], separator="_")

    def algorithm_nick(self) -> str:
        return str_paste(parts=[s.algorithm_nick() for s in self.__selectors], separator="_")

    def uses_covariates(self) -> bool:
        for s in self.__selectors:
            if s.uses_covariates():
                return True
        return False

    def has_fdr_threshold(self) -> bool:
        """Returns false. Even if all descriptors have fdr, they could be different."""
        return False

    def fdr_threshold(self) -> float:
        raise IllegalStateError()

    def __str__(self) -> str:
        return "Many feature selector pipeline with\n" + iterable_to_string(li=self.__selectors, separator="\n") + "\n"


DUMMY_SELECTOR_SINGLE_DESCRIPTOR = DummySingleFeatureSelectorDescriptor()
DUMMY_SELECTOR_MANY_DESCRIPTOR = DummyManyFeatureSelectorDescriptor()
DUMMY_SELECTOR_MO_DESCRIPTOR = DummySelectorMODescriptor()
ANOVA_CATEGORICAL_DESCRIPTOR = AnovaCategoricalDescriptor()
DEFAULT_CATEGORICAL_FS_DESCRIPTOR = ANOVA_CATEGORICAL_DESCRIPTOR
DEFAULT_SURVIVAL_FS_DESCRIPTOR = FeatureSelectorCoxDescriptor()
