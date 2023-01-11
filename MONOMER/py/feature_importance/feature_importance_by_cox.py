from feature_importance.feature_importance import FeatureImportance
from model.survival_model import CoxModel
from util.distribution.distribution import Distribution, ConcreteDistribution


class FeatureImportanceByCox(FeatureImportance):
    """ In our tests with various parameters it was never able to converge."""

    def compute(self, x, y, n_proc: int = 1) -> Distribution:
        model = CoxModel(penalizer=1000.0, step_size=0.2)  # 1.0 0.1 0.01 0.001
        predictor = model.fit(x, y, ignore_warns=False)
        print("predictor")
        print(predictor)
        coefs = predictor.coefficients()
        print("n coeffs: " + str(len(coefs)))
        print("coeffs sum: " + str(sum(coefs)))
        return ConcreteDistribution(probs=coefs)

    def nick(self) -> str:
        return "CoxFI"

    def name(self) -> str:
        return "Cox FI"

    def __str__(self) -> str:
        return "Cox feature importance"
