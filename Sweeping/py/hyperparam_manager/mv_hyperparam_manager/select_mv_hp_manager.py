from hyperparam_manager.mv_hyperparam_manager.abstract_mv_hp_manager import AbstractMvHpManager
from hyperparam_manager.sv_hyperparam_manager.select_hp_manager import SelectHpManager
from hyperparam_manager.view_pops import ViewPops
from util.list_like import ListLike, BoolListLike
from views.adjusted_view_definition import AdjustedViewDef


class SelectMvHpManager(SelectHpManager, AbstractMvHpManager):

    def __init__(self, adj_view_def: AdjustedViewDef, view_pops: ViewPops):
        """View pops must be in alphabetical order of the view names."""
        if adj_view_def.num_predictive_views() != view_pops.num_views():
            raise ValueError()
        view_col_numbers = {
            k: view_pops.view_hyperparams(view_pos=i) for i, k in enumerate(adj_view_def.all_views_seq())}
        SelectHpManager.__init__(self=self, view_pops=view_pops)
        AbstractMvHpManager.__init__(self=self, adj_view_def=adj_view_def, view_col_numbers=view_col_numbers)

    def used_feature_masks(self, hyperparams: ListLike) -> dict[str, BoolListLike]:
        view_individuals = self._view_pops().view_individuals(hyperparams=hyperparams)
        predictive_features: dict[str, BoolListLike] = {}
        for i, k in enumerate(self._adj_view_def().predictive_view_names_seq()):
            ind = view_individuals[i]
            assert isinstance(ind, BoolListLike)
            predictive_features[k] = ind
        return self.add_adjusting_views(current_selection=predictive_features)

    def predictive_features_mask_len_mv(self) -> int:
        return self._view_pops().tot_hyperparams()

    def __str__(self) -> str:
        res = ""
        res += SelectHpManager.__str__(self)
        res += AbstractMvHpManager.__str__(self)
        return res
