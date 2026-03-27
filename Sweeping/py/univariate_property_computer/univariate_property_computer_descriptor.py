from abc import ABC, abstractmethod

from descriptor.descriptor import Descriptor


LOG_UNIVARIATE_NICK = "log"
ANOVA_UNIVARIATE_NICK = "anova"


class UnivariatePropertyComputerDescriptor(Descriptor, ABC):
    pass


class UnivariatePvalComputerDescriptor(UnivariatePropertyComputerDescriptor, ABC):

    @abstractmethod
    def uses_covariates(self) -> bool:
        raise NotImplementedError()

class LogUnivariatePvalComputerDescriptor(UnivariatePvalComputerDescriptor):

    def __str__(self) -> str:
        return "Logistic single feature p-value computer"

    def nick(self) -> str:
        return LOG_UNIVARIATE_NICK

    def uses_covariates(self) -> bool:
        return True


class AnovaUnivariatePvalComputerDescriptor(UnivariatePvalComputerDescriptor):

    def __str__(self) -> str:
        return "ANOVA single feature p-value computer"

    def nick(self) -> str:
        return ANOVA_UNIVARIATE_NICK

    def uses_covariates(self) -> bool:
        return False


LOG_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR = LogUnivariatePvalComputerDescriptor()
ANOVA_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR = AnovaUnivariatePvalComputerDescriptor()
