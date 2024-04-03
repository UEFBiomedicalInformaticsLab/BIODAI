import sklearn
from numpy import ravel, ones

from feature_importance.feature_importance import FeatureImportance
from univariate_feature_selection.univariate_feature_selection import filter_any_nan_mask, filter_low_variance_mask
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.list_math import list_and
from util.utils import p_adjust_bh


class FeatureImportanceAnova(FeatureImportance):

    def compute(self, x, y, n_proc: int = 1) -> Distribution:
        y = ravel(y)
        # Anova does not work with nan or almost zero variance, so we filter out these features,
        # that will get 0 probability in the resulting distribution.
        mask = list_and(filter_any_nan_mask(x), filter_low_variance_mask(x))
        p_values = []
        for i in range(len(mask)):
            if mask[i]:
                anova_res = sklearn.feature_selection.f_classif(X=x[x.columns[[i]]], y=y)
                p_vals = anova_res[1]
                p_values.append(p_vals[0])
        fdr = p_adjust_bh(p_values)
        res = ones(len(mask))
        res[mask] = fdr
        res = 1.0 - res
        return ConcreteDistribution(probs=res)

    def nick(self) -> str:
        return "anovaFI"

    def name(self) -> str:
        return "anova FI"

    def __str__(self) -> str:
        return "anova feature importance"
