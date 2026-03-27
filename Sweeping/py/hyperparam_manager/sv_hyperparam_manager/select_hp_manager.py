from abc import ABC

from hyperparam_manager.hyperparam_manager import HyperparamManager
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from hyperparam_manager.view_pops import ViewPops
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike

from util.feature_space_lifter import FeatureSpaceLifterMV
from util.list_like import ListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet


class SelectHpManager(HyperparamManager, ABC):
    __view_pops: ViewPops

    def __init__(self, view_pops: ViewPops):
        self.__view_pops = view_pops

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        return self.__view_pops.n_predictive_features(hyperparams=hyperparams)

    def max_view_individual_index(self, view_pos: int):
        return self.__view_pops.max_view_individual_index(view_pos=view_pos)

    def __str__(self):
        return str(self.__view_pops)

    def _view_pops(self) -> ViewPops:
        return self.__view_pops


class SelectSvHpManager(SelectHpManager, SvHyperparamManager):

    def __init__(self, view_pops: ViewPops):
        SelectHpManager.__init__(self, view_pops=view_pops)

    def collapsed_used_features_mask(self, hyperparams: ListLike, verbose=False) -> SparseBoolListBySet:
        return self._view_pops().predictive_features_mask(hyperparams=hyperparams, verbose=verbose)

    def predictive_features_mask_len(self, hyperparams: PeculiarIndividualByListlike) -> int:
        return self._view_pops().tot_hyperparams()

    def downlift(self, lifter: FeatureSpaceLifterMV) -> SvHyperparamManager:
        raise NotImplementedError()
