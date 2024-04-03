import warnings
from typing import Sequence, Optional

from pandas import DataFrame
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis

from model.survival_model import SurvivalModel, CoxPredictor, DummyCoxPredictor
from util.dataframes import n_col, scale_df
from util.survival.survival_utils import survival_df_to_sksurv
from util.utils import IllegalStateError


class SksurvModel(SurvivalModel):
    __penalizer: float
    __l1_ratio: float
    __standardize: bool
    __max_iter: int
    __verbose: bool

    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 1.0, standardize: bool = True, max_iter: int = 100000,
                 verbose: bool = False):
        self.__penalizer = penalizer
        self.__l1_ratio = l1_ratio
        self.__standardize = standardize
        self.__max_iter = max_iter
        self.__verbose = verbose

    def fit(self, x: DataFrame, y: DataFrame, sample_weight: Optional = None) -> CoxPredictor:
        """Sample weights are accepted but ignored."""
        if self.__standardize and n_col(x) > 0:
            scaler = StandardScaler().fit(x.values)
            x = scale_df(x, scaler)
        else:
            scaler = None
        y = survival_df_to_sksurv(survival_df=y)
        fitter = CoxnetSurvivalAnalysis(
            alphas=[self.__penalizer], l1_ratio=self.__l1_ratio, normalize=False, copy_X=True, fit_baseline_model=True,
            max_iter=self.__max_iter)
        # fitter = CoxPHSurvivalAnalysis(alpha=alpha)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FitFailedWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                estimator = fitter.fit(x, y)
            return SKSurvCoxPredictor(estimator=estimator, scaler=scaler)
        except BaseException as e:
            if self.__verbose:
                print("Exception caught during fitting of Cox.")
                print(str(e))
            return DummyCoxPredictor(n_coefficients=x.shape[1])

    def nick(self) -> str:
        return "SKSurvCox"

    def name(self) -> str:
        return "SKSurv Cox model"

    def __str__(self) -> str:
        return self.name()


class SKSurvCoxPredictor(CoxPredictor):
    __estimator: CoxnetSurvivalAnalysis
    __scaler: StandardScaler

    def __init__(self, estimator: CoxnetSurvivalAnalysis, scaler: Optional[StandardScaler] = None):
        self.__estimator = estimator
        self.__scaler = scaler

    def coefficients(self) -> Sequence[float]:
        res = self.__estimator.coef_.reshape(-1)
        # print("SKSurv coefficients: " + str(res))
        return res

    def has_p_vals(self) -> bool:
        return False

    def p_vals(self) -> Sequence[float]:
        raise IllegalStateError()

    def score_concordance_index(self, x_test, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        y_test = survival_df_to_sksurv(survival_df=y_test)
        try:
            return self.__estimator.score(X=x_test, y=y_test)
        except ValueError as e:  # SKSurv can raise this exception.
            print("Exception raised by SKSurv while evaluating the c-index:\n" + str(e) + "\n" +
                  "Assigning 0 as c-index.")
            return 0.0

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        raise NotImplementedError()

    def nick(self) -> str:
        return "SKSurvCox"

    def name(self) -> str:
        return "SKSurv Cox predictor"

    def __str__(self) -> str:
        return self.name()
