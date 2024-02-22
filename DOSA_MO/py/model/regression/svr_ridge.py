from typing import Sequence, Optional

from numpy import logspace, ravel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from model.regression.regressor import RegressorModel, Regressor
from model.regression.regressors_library import SVRegressor
from util.math.mean_builder import KahanMeanBuilder
from util.randoms import random_seed
from util.sequence_utils import select_by_indices, reverse


class SVRRidgeModel(RegressorModel):
    __verbosity: int

    def __init__(self, verbosity: int = 1):
        self.__verbosity = verbosity

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> Regressor:
        if self.__verbosity > 0:
            print("Fitting SVR Ridge model.")
        cs = logspace(start=-2, stop=1, num=10)
        epsilons = reverse(logspace(start=-1, stop=0, num=10))  # We prefer larger (more smoothing) epsilons.
        best_mse = None
        best_c = None
        best_epsilon = None
        strata = KFold(n_splits=5, shuffle=True, random_state=random_seed())
        for c in cs:
            for epsilon in epsilons:
                model = SVRegressor(c=c, epsilon=epsilon)
                mean_builder = KahanMeanBuilder()
                for train_index, test_index in strata.split(X=x, y=y):
                    x_train = x.iloc[train_index]
                    y_train = select_by_indices(data=y, indices=train_index)
                    x_test = x.iloc[test_index]
                    y_test = select_by_indices(data=y, indices=test_index)
                    if sample_weight is None:
                        fold_w = None
                        test_w = None
                    else:
                        fold_w = select_by_indices(data=sample_weight, indices=train_index)
                        test_w = select_by_indices(data=sample_weight, indices=test_index)
                    regressor = model.fit(x_train, y_train, sample_weight=fold_w)
                    predictions = regressor.predict(x=x_test)
                    fold_mse = mean_squared_error(
                        y_true=ravel(y_test), y_pred=ravel(predictions), squared=True, sample_weight=test_w)
                    mean_builder.add(fold_mse)
                mse = mean_builder.mean()
                if self.__verbosity > 1:
                    print("c: " + str(c) + " \t" + "epsilon: " + str(epsilon) + " \t" "mse: " + str(mse))
                if best_mse is None or best_mse > mse:  # By using >, we prefer to regularize more.
                    best_mse = mse
                    best_c = c
                    best_epsilon = epsilon
        if self.__verbosity > 0:
            print("best c: " + str(best_c))
            print("best epsilon: " + str(best_epsilon))
            print("best mse: " + str(best_mse))
        return SVRegressor(c=best_c, epsilon=best_epsilon).fit(x, y, sample_weight=sample_weight)

    def nick(self) -> str:
        return "SVRridge"
