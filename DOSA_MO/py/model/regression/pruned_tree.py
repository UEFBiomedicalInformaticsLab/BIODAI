import random
from typing import Sequence, Optional

from numpy import ravel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from model.regression.regressor import RegressorModel, Regressor
from model.regression.regressors_library import TreeRegressor
from util.math.mean_builder import KahanMeanBuilder
from util.randoms import random_seed
from util.sequence_utils import select_by_indices


class PrunedTree(RegressorModel):
    __verbosity: int
    __square_error: bool

    def __init__(self, square_error: bool = False, verbosity: int = 2):
        self.__square_error = square_error
        self.__verbosity = verbosity

    def __criterion(self) -> str:
        if self.__square_error:
            return "squared_error"
        else:
            return "absolute_error"

    def __create_tree(self, alpha=0.0) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(criterion=self.__criterion(), ccp_alpha=alpha, min_samples_leaf=1)

    def fit(self, x, y: Sequence[float], sample_weight: Optional[Sequence[float]] = None) -> Regressor:
        if self.__verbosity > 0:
            print("Fitting pruned tree model.")
        clf = self.__create_tree()
        path = clf.cost_complexity_pruning_path(X=x, y=y, sample_weight=sample_weight)
        alphas = path.ccp_alphas

        best_error = None
        best_alpha = None
        strata = KFold(n_splits=5, shuffle=True, random_state=random_seed())
        rand_state = random.getstate()
        for alpha in alphas:
            random.setstate(rand_state)
            if alpha < 0:
                alpha = 0.0  # Negatives are not accepted.
            model = self.__create_tree(alpha=alpha)
            mean_builder = KahanMeanBuilder()
            for train_index, test_index in strata.split(X=x, y=y):
                x_train = x.iloc[train_index]
                y_train = select_by_indices(data=y, indices=train_index)
                x_test = x.iloc[test_index]
                y_test = select_by_indices(data=y, indices=test_index)
                weight_train = select_by_indices(data=sample_weight, indices=train_index)
                weight_test = select_by_indices(data=sample_weight, indices=test_index)
                regressor = model.fit(x_train, y_train, sample_weight=weight_train)
                predictions = regressor.predict(X=x_test)
                if self.__square_error:
                    fold_error = mean_squared_error(
                            y_true=ravel(y_test), y_pred=ravel(predictions), squared=True, sample_weight=weight_test)
                else:
                    fold_error = mean_absolute_error(
                        y_true=ravel(y_test), y_pred=ravel(predictions), sample_weight=weight_test)
                mean_builder.add(fold_error)
            error = mean_builder.mean()
            if self.__verbosity > 1:
                print("alpha: " + str(alpha) + " \t" "error: " + str(error))
            if best_error is None or best_error > error:
                best_error = error
                best_alpha = alpha
        if self.__verbosity > 0:
            print("best alpha: " + str(best_alpha))
            print("best error: " + str(best_error))
        random.setstate(rand_state)
        return TreeRegressor(
            criterion=self.__criterion(), ccp_alpha=best_alpha, min_samples_leaf=1).fit(
            x, y, sample_weight=sample_weight)

    def nick(self) -> str:
        return "ptree"
