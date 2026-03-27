from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from util import sparse_bool_list_by_set
from util.feature_space_lifter import FeatureSpaceLifterMV
from util.list_like import ListLike, BoolListLike


class DummyHpManager(SvHyperparamManager):

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        return sparse_bool_list_by_set.smart_sum(hyperparams)

    def collapsed_used_features_mask(self, hyperparams: ListLike) -> BoolListLike:
        assert isinstance(hyperparams, BoolListLike)
        return hyperparams

    def predictive_features_mask_len(self, hyperparams: ListLike) -> int:
        return len(hyperparams)

    def downlift(self, lifter: FeatureSpaceLifterMV) -> SvHyperparamManager:
        return self

    def __str__(self) -> str:
        return "dummy hyperparameter manager"


DUMMY_HP_MANAGER = DummyHpManager()