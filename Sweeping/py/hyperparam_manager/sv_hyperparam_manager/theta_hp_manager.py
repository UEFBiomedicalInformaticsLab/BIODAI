from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.list_like import ListLike, BoolListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet


class ThetaHpManager(SvHyperparamManager):
    __theta: float

    def __init__(self, theta: float):
        self.__theta = theta

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        res = 0
        theta = self.__theta
        for h in hyperparams:
            if h > theta:
                res += 1
        return res

    def collapsed_used_features_mask(self, hyperparams: ListLike) -> BoolListLike:
        res = SparseBoolListBySet(min_size=len(hyperparams))
        theta = self.__theta
        for i, h in enumerate(hyperparams):
            if h > theta:
                res.set_true(i)
        return res

    def predictive_features_mask_len(self, hyperparams: ListLike) -> int:
        return len(hyperparams)

    def downlift(self, lifter: FeatureSpaceLifterMV) -> SvHyperparamManager:
        """When downlifted it still works in the same way."""
        return self

    def __str__(self) -> str:
        return "theta hyperparameter manager"
