import random
from collections.abc import Sequence
from typing import Optional

from numpy import ravel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from model.regression.svregressor import RegressorSVModel, SVRegressor
from model.regression.regressors_library import KNRModel
from util.math.mean_builder import KahanMeanBuilder
from util.randoms import random_seed
from util.sequence_utils import reverse
from util.select_from_sequence import select_by_indices


class KnrCvModel(RegressorSVModel):
    __verbosity: int

    def __init__(self, verbosity: int = 2):
        self.__verbosity = verbosity

    def fit(self, x, y: Sequence[float], sample_weight: Optional = None) -> SVRegressor:
        """Weights are ignored."""
        if self.__verbosity > 0:
            print("Fitting KNR CV model.")
        best_mse = None
        best_k = None
        strata = KFold(n_splits=5, shuffle=True, random_state=random_seed())
        max_k = 10
        rand_state = random.getstate()
        for train_index, _ in strata.split(X=x, y=y):
            max_k = min(max_k, len(train_index))  # sklearn complains if k is bigger than number of samples.
        ks = reverse(range(1, max_k+1))  # We prefer larger (more smoothing) ks.
        for k in ks:
            random.setstate(rand_state)
            model = KNRModel(n_neighbors=k)
            mean_builder = KahanMeanBuilder()
            for train_index, test_index in strata.split(X=x, y=y):
                x_train = x.iloc[train_index]
                y_train = select_by_indices(data=y, indices=train_index)
                x_test = x.iloc[test_index]
                y_test = select_by_indices(data=y, indices=test_index)
                regressor = model.fit(x_train, y_train)
                predictions = regressor.predict(x=x_test)
                fold_mse = mean_squared_error(y_true=ravel(y_test), y_pred=ravel(predictions), squared=True)
                mean_builder.add(fold_mse)
            mse = mean_builder.mean()
            if self.__verbosity > 1:
                print("k: " + str(k) + " \t" "mse: " + str(mse))
            if best_mse is None or best_mse > mse:  # By using >, we prefer to regularize more.
                best_mse = mse
                best_k = k
        if self.__verbosity > 0:
            print("best k: " + str(best_k))
            print("best mse: " + str(best_mse))
        random.setstate(rand_state)
        return KNRModel(n_neighbors=best_k).fit(x, y)

    def nick(self) -> str:
        return "KnrCV"
