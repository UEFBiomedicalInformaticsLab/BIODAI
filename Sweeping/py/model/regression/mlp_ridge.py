import random
from typing import Sequence, Optional

from numpy import logspace, ravel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from model.regression.svregressor import RegressorSVModel, SVRegressor
from model.regression.regressors_library import MLPRegressorModel
from util.math.mean_builder import KahanMeanBuilder
from util.randoms import random_seed
from util.sequence_utils import reverse
from util.select_from_sequence import select_by_indices


class MLPRidgeModel(RegressorSVModel):
    __verbosity: int
    __square_error: bool

    def __init__(self, square_error: bool = False, verbosity: int = 2):
        self.__square_error = square_error
        self.__verbosity = verbosity

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> SVRegressor:
        """Sample weights are not used by MLP."""
        if self.__verbosity > 0:
            print("Fitting MLP Ridge model.")
        alphas = reverse(logspace(start=-7, stop=1, num=9))  # We prefer larger (more smoothing) alphas.
        best_mse = None
        best_alpha = None
        strata = KFold(n_splits=5, shuffle=True, random_state=random_seed())
        rand_state = random.getstate()
        for alpha in alphas:
            random.setstate(rand_state)
            model = MLPRegressorModel(alpha=alpha)
            mean_builder = KahanMeanBuilder()
            for train_index, test_index in strata.split(X=x, y=y):
                x_train = x.iloc[train_index]
                y_train = select_by_indices(data=y, indices=train_index)
                x_test = x.iloc[test_index]
                y_test = select_by_indices(data=y, indices=test_index)
                regressor = model.fit(x_train, y_train)
                predictions = regressor.predict(x=x_test)
                if self.__square_error:
                    fold_mse = mean_squared_error(y_true=ravel(y_test), y_pred=ravel(predictions), squared=True)
                else:
                    fold_mse = mean_absolute_error(y_true=ravel(y_test), y_pred=ravel(predictions))
                mean_builder.add(fold_mse)
            mse = mean_builder.mean()
            if self.__verbosity > 1:
                print("alpha: " + str(alpha) + " \t" "mse: " + str(mse))
            if best_mse is None or best_mse > mse:  # By using >, we prefer to regularize more.
                best_mse = mse
                best_alpha = alpha
        if self.__verbosity > 0:
            print("best alpha: " + str(best_alpha))
            print("best mse: " + str(best_mse))
        random.setstate(rand_state)
        return MLPRegressorModel(alpha=best_alpha).fit(x, y)

    def nick(self) -> str:
        return "MLPridge"
