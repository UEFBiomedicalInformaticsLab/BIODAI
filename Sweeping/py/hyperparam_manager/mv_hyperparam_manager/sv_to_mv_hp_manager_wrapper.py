from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from hyperparam_manager.sv_hyperparam_manager.sv_hyperparam_manager import SvHyperparamManager
from util.list_like import ListLike, BoolListLike
if TYPE_CHECKING:
    from individual.peculiar_individual_with_context_mv import PeculiarIndividualWithContextMV
    from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV


class SvToMvHpManagerWrapper(MvHyperparamManager):
    """This kind of MV HP manager wraps a single view manager and handles the features as collapsed in a single view."""
    __sv_hp_manager : SvHyperparamManager
    __view_name: str
    __predictive_features_num: int

    def __init__(self, sv_hp_manager: SvHyperparamManager, view_name: str, predictive_features_num: int):
        """predictive_features_num is the total number of existing predictive features.
        This must be the same as the number of all kinds of features, since adjusting features are not supported
        by this HP manager."""
        self.__sv_hp_manager = sv_hp_manager
        self.__view_name = view_name
        self.__predictive_features_num = predictive_features_num

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        return self.__sv_hp_manager.n_predictive_features(hyperparams=hyperparams)

    def used_feature_masks(self, hyperparams: ListLike) -> dict[str, BoolListLike]:
        mask = self.__sv_hp_manager.collapsed_used_features_mask(hyperparams=hyperparams)
        return {self.__view_name: mask}

    def predictive_features_mask_len_mv(self) -> int:
        return self.__predictive_features_num

    def  contextualize_all(
            self, pop: Iterable[ConfidentPredictiveIndividualMV]) -> Sequence[PeculiarIndividualWithContextMV]:
        from individual.peculiar_individual_with_context_mv import PeculiarIndividualWithContextMV
        res = []
        for individual in pop:
            res.append(PeculiarIndividualWithContextMV(individual=individual, hp_manager=self))
        return res
