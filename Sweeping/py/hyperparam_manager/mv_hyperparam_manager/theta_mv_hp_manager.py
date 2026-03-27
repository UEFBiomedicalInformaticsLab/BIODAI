from hyperparam_manager.mv_hyperparam_manager.abstract_mv_hp_manager import AbstractMvHpManager
from util.list_like import ListLike, BoolListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet
from views.adjusted_view_definition import AdjustedViewDef


class ThetaMvHpManager(AbstractMvHpManager):
    __theta: float

    def __init__(self, adj_view_def: AdjustedViewDef, view_col_numbers: dict[str, int], theta: float):
        AbstractMvHpManager.__init__(self=self, adj_view_def=adj_view_def, view_col_numbers=view_col_numbers)
        self.__theta = theta

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        if len(hyperparams) == self.predictive_features_mask_len_mv():
            res = 0
            theta = self.__theta
            for h in hyperparams:
                if h > theta:
                    res += 1
            return res
        else:
            raise ValueError()

    def used_feature_masks(self, hyperparams: ListLike) -> dict[str, BoolListLike]:
        bool_hp = SparseBoolListBySet(min_size=len(hyperparams))
        theta = self.__theta
        for i, h in enumerate(hyperparams):
            if h > theta:
                bool_hp.set_true(i)
        predictive_features = self._predictive_views_to_concat().concat_to_mv_masks(concat_mask=bool_hp)
        return self.add_adjusting_views(current_selection=predictive_features)

    def __str__(self) -> str:
        return "theta multi-view hyperparameter manager"