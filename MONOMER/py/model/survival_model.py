import warnings
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from lifelines import CoxPHFitter
from pandas import Series
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from model.model import Predictor, Model
from util.dataframes import scale_df, n_col, non_finite_report
from util.named import NickNamed
from util.summer import KahanSummer
from util.utils import IllegalStateError, null_showwarning


SURVIVAL_DURATION_STR = 'duration'
SURVIVAL_EVENT_STR = 'event'


class SurvivalPredictor(Predictor, NickNamed, ABC):

    def is_class_predictor(self) -> bool:
        return False

    def predict(self, x):
        raise IllegalStateError()


class SurvivalModel(Model, ABC):

    def is_class_model(self) -> bool:
        return False


def cox_merge_x_y(x, y):
    # TODO: Drop row labels if present.
    events = pd.Series(y[:][SURVIVAL_EVENT_STR], name=SURVIVAL_EVENT_STR)
    times = pd.Series(y[:][SURVIVAL_DURATION_STR], name=SURVIVAL_DURATION_STR)
    res = pd.concat([x, events, times], axis=1)
    return res


class CoxPredictor(SurvivalPredictor, ABC):

    @abstractmethod
    def coefficients(self) -> [float]:
        raise NotImplementedError()


class ProperCoxPredictor(CoxPredictor):
    __estimator: CoxPHFitter
    __scaler: StandardScaler

    def __init__(self, estimator: CoxPHFitter, scaler: Optional[StandardScaler] = None):
        self.__scaler = scaler
        self.__estimator = estimator

    def score_concordance_index(self, x_test, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        df = cox_merge_x_y(x=x_test, y=y_test)
        return self.__estimator.score(df, scoring_method="concordance_index")

    def score_log_likelihood(self, x_test, y_test) -> float:
        if self.__scaler is not None:
            x_test = scale_df(x_test, self.__scaler)
        df = cox_merge_x_y(x=x_test, y=y_test)
        return self.__estimator.score(df, scoring_method="log_likelihood")

    def p_vals(self):
        summary = self.__estimator.summary
        return summary['p']

    def coefficients(self) -> Series:
        return self.__estimator.params_

    def nick(self) -> str:
        return "Cox"

    def __str__(self) -> str:
        return "proper Cox predictor"


class DummyCoxPredictor(CoxPredictor):
    __n_coefficients: int

    def __init__(self, n_coefficients: int):
        self.__n_coefficients = n_coefficients

    def score_concordance_index(self, x_test, y_test) -> float:
        return 0.0

    def nick(self) -> str:
        return "dummy Cox"

    def __str__(self) -> str:
        return "dummy Cox predictor"

    def coefficients(self) -> [float]:
        return [0.0] * self.__n_coefficients


class CoxModel(SurvivalModel):
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
    def fit(self, x, y, ignore_warns: bool = True) -> CoxPredictor:
        if self.__standardize and n_col(x) > 0:
            scaler = StandardScaler().fit(x)
            x = scale_df(x, scaler)
        else:
            scaler = None
        df = cox_merge_x_y(x=x, y=y)
        fitter = CoxPHFitter(penalizer=self.__penalizer, l1_ratio=self.__l1_ratio)
        try:
            if ignore_warns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    warnings.showwarning = null_showwarning
                    estimator = fitter.fit(df=df, duration_col=SURVIVAL_DURATION_STR,
                                           event_col=SURVIVAL_EVENT_STR, show_progress=False,
                                           step_size=self.__step_size)
            else:
                estimator = fitter.fit(df=df, duration_col=SURVIVAL_DURATION_STR,
                                       event_col=SURVIVAL_EVENT_STR, show_progress=False,
                                       step_size=self.__step_size)
            return ProperCoxPredictor(estimator=estimator, scaler=scaler)
        except BaseException as e:
            if self.__verbose:
                print("Exception caught during fitting of Cox.")
                print(str(e))
                print(non_finite_report(df))
            return DummyCoxPredictor(n_coefficients=x.shape[1])

    def nick(self) -> str:
        return "Cox"

    def name(self) -> str:
        return self.nick()

    def __str__(self) -> str:
        return self.name()


def create_folds(x, y, n_folds: int = 10, seed=4985):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = []
    y_event = y[[SURVIVAL_EVENT_STR]]
    for train_index, test_index in skf.split(X=x, y=y_event):
        res.append([train_index, test_index])
    return res


def train_test_one_fold(x_train, y_train, x_test, y_test, model: SurvivalModel):
    """ Returns the concordance index. """
    predictor = model.fit(x=x_train, y=y_train)
    score = predictor.score_concordance_index(x_test=x_test, y_test=y_test)
    return score


def cross_validate(x, y, model: SurvivalModel, n_folds: int = 10, seed=78245):
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
