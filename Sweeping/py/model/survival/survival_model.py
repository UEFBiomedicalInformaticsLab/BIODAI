import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import pandas as pd

from lifelines import CoxPHFitter
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from model.sv_model import SVPredictor, SVModel
from util.dataframe.dataframes import scale_df, non_finite_report, create_df_with_repeated_value
from util.table.table_utils import n_col, n_row
from util.named import NickNamed
from util.math.summer import KahanSummer
from util.survival.survival_utils import SURVIVAL_DURATION_STR, SURVIVAL_EVENT_STR
from util.utils import IllegalStateError, null_showwarning


COX_NICK = "Cox"


class SurvivalSVPredictor(SVPredictor, NickNamed, ABC):

    def predict(self, x: DataFrame) -> Sequence:
        raise IllegalStateError()

    def predict_crisp(self, x: DataFrame) -> Sequence:
        raise IllegalStateError()


class SurvivalSVModel(SVModel, ABC):
    pass


def cox_merge_x_y(x: Union[DataFrame, ndarray], y: DataFrame):
    """Ignores existing row labels."""
    if not isinstance(x, DataFrame):
        x = DataFrame(x)
    events = pd.Series(y[:][SURVIVAL_EVENT_STR], name=SURVIVAL_EVENT_STR)
    events.reset_index(drop=True, inplace=True)
    times = pd.Series(y[:][SURVIVAL_DURATION_STR], name=SURVIVAL_DURATION_STR)
    times.reset_index(drop=True, inplace=True)
    x = x.reset_index(drop=True, inplace=False)
    return pd.concat([x, events, times], axis=1)


class CoxPredictor(SurvivalSVPredictor, ABC):

    @abstractmethod
    def coefficients(self) -> Sequence[float]:
        raise NotImplementedError()

    @abstractmethod
    def has_p_vals(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def p_vals(self) -> Sequence[float]:
        raise NotImplementedError()


class LifelinesPredictor(CoxPredictor):
    __estimator: CoxPHFitter
    __scaler: StandardScaler
    __imputer: SimpleImputer

    def __init__(self, estimator: CoxPHFitter, scaler: Optional[StandardScaler] = None,
                 imputer: Optional[SimpleImputer] = None):
        self.__scaler = scaler
        self.__estimator = estimator
        self.__imputer = imputer

    def score_concordance_index(self, x_test: DataFrame, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        if self.__imputer is not None:
            x_test = self.__imputer.transform(x_test)
        df = cox_merge_x_y(x=x_test, y=y_test)
        return self.__estimator.score(df, scoring_method="concordance_index")

    def score_log_likelihood(self, x_test: DataFrame, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        if self.__imputer is not None:
            x_test = self.__imputer.transform(x_test)
        df = cox_merge_x_y(x=x_test, y=y_test)
        return self.__estimator.score(df, scoring_method="log_likelihood")

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        if self.__scaler is not None:
            x = scale_df(x, self.__scaler)
        if self.__imputer is not None:
            x = self.__imputer.transform(x)
        return self.__estimator.predict_survival_function(X=x, times=times)

    def has_p_vals(self) -> bool:
        return True

    def p_vals(self) -> Sequence[float]:
        """Probably computed using Wald test."""
        summary = self.__estimator.summary
        return list(summary['p'])

    def coefficients(self) -> Series:
        return self.__estimator.params_

    def nick(self) -> str:
        return "LifelinesCox"

    def __str__(self) -> str:
        return "lifelines Cox predictor"


class DummyCoxPredictor(CoxPredictor):

    __n_coefficients: int

    def __init__(self, n_coefficients: int):
        self.__n_coefficients = n_coefficients

    def score_concordance_index(self, x_test, y_test) -> float:
        return 0.5

    def nick(self) -> str:
        return "DummyCox"

    def __str__(self) -> str:
        return "dummy Cox predictor"

    def coefficients(self) -> Sequence[float]:
        return [0.0] * self.__n_coefficients

    def predict_survival_probabilities(self, x: DataFrame, times: Sequence[float]) -> DataFrame:
        return create_df_with_repeated_value(value=0.5, height=len(times), width=n_row(x))

    def has_p_vals(self):
        return False

    def p_vals(self):
        raise IllegalStateError()


class LifelinesModel(SurvivalSVModel):
    """Probability for specific subject at specific times: predict_survival_function or
    predict_cumulative_hazard.
    For Brier scores perhaps sksurv.metrics.integrated_brier_score could do (looking at source code, it computes
    the censoring distribution on training data, perhaps we can pass test data there, in fact they do that in
    an example: integrated_brier_score(y, y, preds, times)).
    integrated_brier_score gives equal weight to equal time duration.
    There is also pysurvival.utils.metrics.integrated_brier_score.
    It does the same, but wants a model of its framework.
    Only Brier score at a time point: sksurv.metrics.brier_score"""
    __penalizer: float
    __l1_ratio: float
    __step_size: float
    __standardize: bool
    __verbose: bool

    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 1.0, step_size: float = 0.9,
                 standardize: bool = True, verbose: bool = False):
        self.__penalizer = penalizer
        self.__l1_ratio = l1_ratio
        self.__step_size = step_size
        self.__standardize = standardize
        self.__verbose = verbose

    # @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x: DataFrame, y: DataFrame, sample_weights: Optional = None, ignore_warns: bool = True
            ) -> CoxPredictor:
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
        df = cox_merge_x_y(x=x, y=y)
        fitter = CoxPHFitter(penalizer=self.__penalizer, l1_ratio=self.__l1_ratio)  # Does not accept NaN values.
        try:
            if ignore_warns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    warnings.showwarning = null_showwarning
                    estimator = fitter.fit(df=df, duration_col=SURVIVAL_DURATION_STR,
                                           event_col=SURVIVAL_EVENT_STR, show_progress=False,
                                           fit_options={"step_size": self.__step_size})
            else:
                estimator = fitter.fit(df=df, duration_col=SURVIVAL_DURATION_STR,
                                       event_col=SURVIVAL_EVENT_STR, show_progress=False,
                                       fit_options={"step_size": self.__step_size})
            return LifelinesPredictor(estimator=estimator, scaler=scaler, imputer=imputer)
        except BaseException as e:
            if self.__verbose:
                print("Exception caught during fitting of Cox.")
                print(str(e))
                print(non_finite_report(df))
            return DummyCoxPredictor(n_coefficients=x.shape[1])

    def nick(self) -> str:
        return "LifelinesCox"

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()


def create_folds(x: DataFrame, y, n_folds: int = 10, seed=4985):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = []
    y_event = y[[SURVIVAL_EVENT_STR]]
    for train_index, test_index in skf.split(X=x, y=y_event):
        res.append([train_index, test_index])
    return res


def train_test_one_fold(x_train: DataFrame, y_train, x_test: DataFrame, y_test, model: SurvivalSVModel):
    """ Returns the concordance index. """
    predictor = model.fit(x=x_train, y=y_train)
    score = predictor.score_concordance_index(x_test=x_test, y_test=y_test)
    return score


def cross_validate(x: DataFrame, y, model: SurvivalSVModel, n_folds: int = 10, seed=78245):
    folds = create_folds(x, y, n_folds=n_folds, seed=seed)
    scores = []
    for train_index, test_index in folds:
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        fold_score = train_test_one_fold(x_train, y_train, x_test, y_test, model=model)
        scores.append(fold_score)
    return KahanSummer.mean(scores)
