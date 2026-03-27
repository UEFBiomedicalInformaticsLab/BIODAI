from __future__ import annotations

from hyperparam_manager.mv_hyperparam_manager.abstract_mv_hp_manager import AbstractMvHpManager
from input_data.input_data import InputData
from util.list_like import ListLike, BoolListLike
from views.adjusted_view_definition import AdjustedViewDef


class MaskMvHpManager(AbstractMvHpManager):
    """The hyperparams are a mask of used predictive features."""

    def __init__(self, adj_view_def: AdjustedViewDef, view_col_numbers: dict[str, int]):
        """view_col_numbers must contain both predictive and adjusting views."""
        AbstractMvHpManager.__init__(self=self, adj_view_def=adj_view_def, view_col_numbers=view_col_numbers)

    def n_predictive_features(self, hyperparams: ListLike):
        if len(hyperparams) == self.predictive_features_mask_len_mv():
            return hyperparams.sum()
        else:
            raise ValueError()

    def used_feature_masks(self, hyperparams: ListLike) -> dict[str, BoolListLike]:
        predictive_features = self._predictive_views_to_concat().concat_to_mv_masks(concat_mask=hyperparams)
        return self.add_adjusting_views(current_selection=predictive_features)

    def __str__(self) -> str:
        return "feature mask multi-view hyperparameter manager\n" + super().__str__()

    @staticmethod
    def create_from_input_data(input_data: InputData) -> MaskMvHpManager:
        return MaskMvHpManager(
            adj_view_def=input_data.adjusted_view_def(), view_col_numbers=input_data.n_features_per_view())