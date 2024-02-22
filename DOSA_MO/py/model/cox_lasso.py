import warnings
from typing import Optional

import pandas as pd
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis

from model.model import Predictor
from model.sksurv_model import SksurvModel
from model.survival_model import SurvivalModel
from util.dataframes import n_col, scale_df
from util.randoms import random_seed
from util.survival.survival_utils import survival_df_to_sksurv


class CoxLasso(SurvivalModel):
    __l1_ratio: float
    __standardize: bool
    __max_iter: int
    __verbose: bool

    def __init__(self, l1_ratio: float = 1.0, standardize: bool = True, max_iter: int = 100, verbose: bool = False):
        self.__l1_ratio = l1_ratio
        self.__standardize = standardize
        self.__max_iter = max_iter
        self.__verbose = verbose

    def fit(self, x, y, sample_weight: Optional=None) -> Predictor:
        """Sample weights are ignored at the moment."""
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=(1.0/3.0), shuffle=True, random_state=random_seed())
        y_train = pd.DataFrame(y_train, columns=y.columns)
        y_test = pd.DataFrame(y_test, columns=y.columns)
        if self.__standardize and n_col(x) > 0:
            train_scaler = StandardScaler().fit(x_train.values)
            x_train = scale_df(x_train, train_scaler)
            x_test = scale_df(x_test, train_scaler)

        fitter = CoxnetSurvivalAnalysis(
            l1_ratio=self.__l1_ratio, alpha_min_ratio=0.01, normalize=False, copy_X=True, fit_baseline_model=True,
            max_iter=self.__max_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FitFailedWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fitter.fit(x_train, survival_df_to_sksurv(survival_df=y_train))
        estimated_alphas = fitter.alphas_
        best_alpha = 0.0
        best_c = 0
        for alpha in estimated_alphas:
            temp_model = SksurvModel(penalizer=alpha, l1_ratio=self.__l1_ratio, standardize=False,
                                     max_iter=self.__max_iter, verbose=False)
            temp_predictor = temp_model.fit(x=x_train, y=y_train)
            temp_c = temp_predictor.score_concordance_index(x_test=x_test, y_test=y_test)
            if temp_c > best_c:
                best_alpha = alpha
                best_c = temp_c
        best_model = SksurvModel(penalizer=best_alpha, l1_ratio=self.__l1_ratio, standardize=False,
                                 max_iter=self.__max_iter, verbose=False)
        return best_model.fit(x, y)

    def nick(self) -> str:
        return "CoxLasso"

    def name(self) -> str:
        return "Cox LASSO"

    def __str__(self) -> str:
        return "Cox LASSO model"
