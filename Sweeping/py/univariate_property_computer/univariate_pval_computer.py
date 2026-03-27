import math
import warnings
from abc import abstractmethod, ABC
from typing import Sequence, Optional

import pandas as pd
from numpy import number, ravel
from pandas import Series

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import HessianInversionWarning

from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from univariate_feature_selection.parallel_anova import anova_pval_one_feature_checked
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from univariate_property_computer.univariate_property_computer_descriptor import UnivariatePvalComputerDescriptor, \
    LOG_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR, ANOVA_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR
from util.table.table import Table


class UnivariatePvalComputer(UnivariatePropertyComputer, ABC):

    @abstractmethod
    def inner_compute_property(
            self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None) -> float:
        """Covariates must not contain NaN or inf.
        Returns NaN if it is not possible to compute the p-value (e.g. no convergence)."""
        raise NotImplementedError()

    def ignores_covariates(self) -> bool:
        return not self.descriptor().uses_covariates()

    def descriptor(self) -> UnivariatePvalComputerDescriptor:
        res = UnivariatePropertyComputer.descriptor(self=self)
        assert isinstance(res, UnivariatePvalComputerDescriptor)
        return res


class LogUnivariatePvalComputer(UnivariatePvalComputer):
    __skip_hessian: bool
    __maxiter: int
    __method: str

    def __init__(self, skip_hessian: bool = False, maxiter: int = 35, method: str = 'newton'):
        """If Hessian is skipped all methods except Newton do not compute the p-values.
        The Newton method is the only one that in tests was consistently converging when used with 1 SNP + clinical
        data, even when maxiter was increased to 1000."""
        UnivariatePvalComputer.__init__(self=self, descriptor=LOG_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR)
        self.__skip_hessian = skip_hessian
        self.__maxiter = maxiter
        self.__method = method

    def inner_compute_property(self, feature: Sequence[number], outcome: Outcome,
                               covariates: Optional[Table] = None) -> float:

        # Generate a unique column name
        base_name = "x"
        new_col_name = base_name

        if covariates is None:
            covariates_df = pd.DataFrame({new_col_name: Series(feature, name=new_col_name)})
        else:
            covariates_df = covariates.to_dataframe()
            covariates_df.reset_index(inplace=True, drop=True)

            counter = 1
            while new_col_name in covariates_df.columns:
                new_col_name = f"{base_name}_{counter}"
                counter += 1

            # Convert the sequence to a pandas Series
            new_column_df = pd.DataFrame({new_col_name: Series(feature, name=new_col_name)})

            # Insert the new column at the beginning of the DataFrame
            covariates_df = pd.concat([new_column_df,covariates_df], axis=1)

        # Add a constant to the model (intercept)
        covariates_df = sm.add_constant(covariates_df)
        # Encode the categorical variable using pd.factorize()
        y, _ = pd.factorize(ravel(outcome.data()))
        # Normalize the encoded values to be within the unit interval [0, 1] as required by statsmodels.
        y = y / y.max()
        # statsmodels is used because sklearn does not offer p-values.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if not self.__skip_hessian:
                warnings.simplefilter('ignore', HessianInversionWarning)
            try:
                model = sm.Logit(y, covariates_df).fit(
                    method=self.__method, disp=False, skip_hessian=self.__skip_hessian, warn_convergence=False,
                    maxiter=self.__maxiter)
            except BaseException:
                return math.nan
        pvalue = model.pvalues[new_col_name]
        return pvalue

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,


class AnovaUnivariatePvalComputer(UnivariatePvalComputer):

    def __init__(self):
        UnivariatePvalComputer.__init__(self=self, descriptor=ANOVA_UNIVARIATE_PVAL_COMPUTER_DESCRIPTOR)

    def inner_compute_property(self, feature: Sequence[number], outcome: Outcome,
                               covariates: Optional[Table] = None) -> float:

        return anova_pval_one_feature_checked(x=feature, y=outcome.first_col(), verbose=False, ignore_warns=True)

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.categorical,

    def ignores_covariates(self) -> bool:
        return False
