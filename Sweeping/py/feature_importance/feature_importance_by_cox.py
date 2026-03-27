from pandas import DataFrame

from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome
from model.survival.cox_lasso import CoxLasso
from model.survival.survival_model import LifelinesModel
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.list_math import num_of_nonzero, list_abs
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table


USE_SKSURV = True


class FeatureImportanceByCox(FeatureImportance):
    """Warning: do not use with huge datasets, it trains on all the features."""

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        return self.compute_df(x=x.to_dataframe(), y=y.data(), n_proc=n_proc, printer=printer)

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        if USE_SKSURV:
            model = CoxLasso()  # SksurvModel(penalizer=0.03, max_iter=500000)
            predictor = model.fit(x, y)
        else:
            # In our tests with various parameters it was never able to converge.
            model = LifelinesModel(penalizer=1000.0, step_size=0.2)  # 1.0 0.1 0.01 0.001
            predictor = model.fit(x, y, ignore_warns=False)
        signed_coefs = predictor.coefficients()
        coefs = list_abs(signed_coefs)
        printer.print("predictor")
        printer.print("predictor")
        printer.print(predictor)
        printer.print("n coeffs: " + str(len(coefs)))
        printer.print("n non-zero coeffs: " + str(num_of_nonzero(coefs)))
        printer.print("coeffs sum: " + str(sum(coefs)))
        return ConcreteDistribution(probs=coefs)

    def nick(self) -> str:
        return "CoxFI"

    def name(self) -> str:
        return "Cox FI"

    def __str__(self) -> str:
        return "Cox feature importance"
