from typing import Sequence, Optional

from numpy import ravel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from model.regression.regressor import RegressorModel, Regressor
from model.regression.regressors_library import SVRegressor
from util.math.mean_builder import KahanMeanBuilder
from util.randoms import random_seed, log10_random
from util.sequence_utils import select_by_indices


class RandomizedSVRModel(RegressorModel):
    __verbosity: int

    def __init__(self, verbosity: int = 1):
        self.__verbosity = verbosity

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> Regressor:
        if self.__verbosity > 0:
            print("Fitting randomized SVR model.")
        num_extractions = 1000
        cs = [log10_random(min_val=0.01, max_val=100.0) for _ in range(num_extractions)]
        epsilons = [log10_random(min_val=0.001, max_val=10.0) for _ in range(num_extractions)]
        best_mae = None
        best_c = None
        best_epsilon = None
        strata = KFold(n_splits=5, shuffle=True, random_state=random_seed())
        for c, epsilon in zip(cs, epsilons):
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
                fold_mse = mean_absolute_error(
                    y_true=ravel(y_test), y_pred=ravel(predictions), sample_weight=test_w)
                mean_builder.add(fold_mse)
            mae = mean_builder.mean()
            if self.__verbosity > 1:
                print("c: " + str(c) + " \t" + "epsilon: " + str(epsilon) + " \t" "mae: " + str(mae))
            if best_mae is None or best_mae > mae:
                best_mae = mae
                best_c = c
                best_epsilon = epsilon
        if self.__verbosity > 0:
            print("best c: " + str(best_c))
            print("best epsilon: " + str(best_epsilon))
            print("best mae: " + str(best_mae))
        return SVRegressor(c=best_c, epsilon=best_epsilon).fit(x, y, sample_weight=sample_weight)

    def nick(self) -> str:
        return "rSVR"
