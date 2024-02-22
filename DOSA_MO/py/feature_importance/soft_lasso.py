from feature_importance.feature_importance_average import FeatureImportanceAverage
from feature_importance.feature_importance_by_lasso import FeatureImportanceByLasso
from feature_importance.feature_importance_uniform import FeatureImportanceUniform


class SoftLasso(FeatureImportanceAverage):

    def __init__(self):
        FeatureImportanceAverage.__init__(self, [FeatureImportanceUniform(), FeatureImportanceByLasso()])

    def nick(self) -> str:
        return "SoftLassoFI"

    def name(self) -> str:
        return "soft lasso FI"

    def __str__(self) -> str:
        return "soft lasso feature importance"
