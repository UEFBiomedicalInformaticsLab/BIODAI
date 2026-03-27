from collections.abc import Sequence

from numpy import ravel
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.summer import KahanSummer
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table
from util.table.table_utils import n_col


def collapse_coef(coef: Sequence[Sequence[float]]) -> Sequence[float]:
    """Collapses coefficients for multiple classes to a single sequence."""
    n_features = len(coef[0])
    n_classes = len(coef)
    res = []
    for i in range(n_features):
        feat_summer = KahanSummer()
        for j in range(n_classes):
            feat_summer.add(abs(coef[j][i]))
        res.append(feat_summer.get_sum())
    return res


class FeatureImportanceByLasso(FeatureImportance):
    """Warning: do not use with huge datasets, it trains on all the features."""

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        return self.compute_df(x=x.to_dataframe(), y=y.data(), n_proc=n_proc, printer=printer)

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        if n_col(x) == 0:
            return ConcreteDistribution(probs=[])
        y = ravel(y)
        imputer = SimpleImputer()
        logistic_reg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        pipe = make_pipeline(imputer, StandardScaler(), logistic_reg)
        pipe.fit(x.values, y)
        coefs = logistic_reg.coef_
        res = collapse_coef(coef=coefs)
        return ConcreteDistribution(probs=res)

    def nick(self) -> str:
        return "lassoFI"

    def name(self) -> str:
        return "lasso FI"

    def __str__(self) -> str:
        return "lasso feature importance"
