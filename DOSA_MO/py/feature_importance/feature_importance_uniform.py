from feature_importance.feature_importance import FeatureImportance
from util.distribution.uniform_distribution import UniformDistribution


class FeatureImportanceUniform(FeatureImportance):

    def compute(self, x, y=None, n_proc: int = 1) -> UniformDistribution:
        return UniformDistribution(size=x.shape[1])  # Works for pandas and numpy

    def nick(self) -> str:
        return "uniformFI"

    def name(self) -> str:
        return "uniform FI"

    def __str__(self) -> str:
        return "uniform feature importance"
