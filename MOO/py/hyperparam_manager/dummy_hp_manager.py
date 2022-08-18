from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from hyperparam_manager.hyperparam_manager import HyperparamManager
from util import sparse_bool_list_by_set
from util.list_like import ListLike


class DummyHpManager(HyperparamManager):

    @staticmethod
    def n_active_features(hyperparams):
        return sparse_bool_list_by_set.smart_sum(hyperparams)

    def active_features_mask(self, hyperparams: ListLike) -> ListLike:
        return hyperparams

    def active_features_mask_len(self, hyperparams) -> int:
        return len(hyperparams)

    @staticmethod
    def active_features_mask_numpy(hyperparams: PeculiarIndividualByListlike):
        return hyperparams.to_numpy()

    def __str__(self) -> str:
        return "dummy hyperparameter manager"
