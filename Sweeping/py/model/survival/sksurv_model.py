import warnings
from typing import Sequence, Optional

from pandas import DataFrame
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis

from model.survival.survival_model import SurvivalSVModel, CoxPredictor, DummyCoxPredictor
from util.dataframe.dataframes import scale_df
from util.table.table_utils import n_col
from util.survival.survival_utils import survival_df_to_sksurv
from util.utils import IllegalStateError


class SksurvModel(SurvivalSVModel):
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
        scaler = None
        imputer = None
        if n_col(x) > 0:
            if self.__standardize:
                scaler = StandardScaler().fit(x.values)
                x = scale_df(x, scaler)
            imputer = SimpleImputer()
            imputer.fit(x)
            x = imputer.transform(x)
        y = survival_df_to_sksurv(survival_df=y)
        fitter = CoxnetSurvivalAnalysis(
            alphas=[self.__penalizer], l1_ratio=self.__l1_ratio, normalize=False, copy_X=True, fit_baseline_model=True,
            max_iter=self.__max_iter)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FitFailedWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                estimator = fitter.fit(x, y)
            return SKSurvCoxPredictor(estimator=estimator, scaler=scaler, imputer=imputer)
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
    __imputer: SimpleImputer

    def __init__(self, estimator: CoxnetSurvivalAnalysis, scaler: Optional[StandardScaler] = None,
                 imputer: Optional[SimpleImputer] = None):
        self.__estimator = estimator
        self.__scaler = scaler
        self.__imputer = imputer

    def coefficients(self) -> Sequence[float]:
        res = self.__estimator.coef_.reshape(-1)
        # print("SKSurv coefficients: " + str(res))
        return res

    def has_p_vals(self) -> bool:
        return False

    def p_vals(self) -> Sequence[float]:
        raise IllegalStateError()

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        if self.__imputer is not None:
            x_test = self.__imputer.transform(x_test)
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
