from feature_importance.feature_importance import FeatureImportance
from model.cox_lasso import CoxLasso
from model.survival_model import LifelinesModel
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.math.list_math import num_of_nonzero, list_abs


class FeatureImportanceByCox(FeatureImportance):

    def compute(self, x, y, n_proc: int = 1, verbose: bool = True) -> Distribution:
        USE_SKSURV = True
        if USE_SKSURV:
            model = CoxLasso() # SksurvModel(penalizer=0.03, max_iter=500000)
            predictor = model.fit(x, y)
        else:
            # In our tests with various parameters it was never able to converge.
            model = LifelinesModel(penalizer=1000.0, step_size=0.2)  # 1.0 0.1 0.01 0.001
            predictor = model.fit(x, y, ignore_warns=False)
        signed_coefs = predictor.coefficients()
        coefs = list_abs(signed_coefs)
        if verbose:
            print("predictor")
            print(predictor)
            print("n coeffs: " + str(len(coefs)))
            print("n non-zero coeffs: " + str(num_of_nonzero(coefs)))
            print("coeffs sum: " + str(sum(coefs)))
        return ConcreteDistribution(probs=coefs)

    def nick(self) -> str:
        return "CoxFI"

    def name(self) -> str:
        return "Cox FI"

    def __str__(self) -> str:
        return "Cox feature importance"
