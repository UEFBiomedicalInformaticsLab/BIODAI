from __future__ import annotations
from abc import ABC
from collections.abc import Sequence, Iterable
from typing import Optional, TYPE_CHECKING

from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from hyperparam_manager.mv_hyperparam_manager.mv_to_concat_mapper import MvToConcatMapper
from util.dict_utils import sorted_dict
from util.list_like import BoolListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet, union_of_pair_sparse, SparseBoolList
from util.uniform_list import BoolUniformList
from util.utils import PlannedUnreachableCodeError
from views.adjusted_view_definition import AdjustedViewDef
if TYPE_CHECKING:
    from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV
    from individual.peculiar_individual_with_context_mv import PeculiarIndividualWithContextMV


def add_adjusting_views(current_selection: dict[str, Sequence[bool]],
                        adj_view_def: AdjustedViewDef,
                        view_sizes: Optional[dict[str, int]] = None) -> dict[str, SparseBoolList]:
    """Returned dict is sorted by keys. Views with 0 active features are still included in the result.
    Adjusting views that adjust predictive views with zero features are included, but with zero features,
    unless that adjusting view is specified as having selected features in the current_selection.
    If view_sizes is not provided, it is assumed that all necessary views are included in current_selection."""
    if view_sizes is None:
        view_sizes = {k: len(v) for k, v in current_selection.items()}
    res: dict[str, SparseBoolList] = {v: SparseBoolListBySet(min_size=view_sizes[v])
                                      for v in adj_view_def.all_views_seq()}
    for k, v in current_selection.items():
        v = SparseBoolListBySet(seq=v)
        if adj_view_def.is_predictive_view(view_name=k):
            res[k] = v
            if v.sum() > 0:  # Activate all features of adjusting views only if there is at least an active feature.
                for adj_k in adj_view_def.adjusters_for_view(view=k):
                    res[adj_k] = SparseBoolListBySet(
                        seq=BoolUniformList(value=True, size=view_sizes[adj_k]))
        elif adj_view_def.is_adjusting_view(view_name=k):
            res[k] = union_of_pair_sparse(list_a=res[k], list_b=v)
        else:
            raise PlannedUnreachableCodeError()
    return sorted_dict(d=res)


class AbstractMvHpManager(MvHyperparamManager, ABC):
    __adj_view_def: AdjustedViewDef
    __predictive_views_to_concat: MvToConcatMapper
    __predictive_features_num: int
    __view_col_numbers: dict[str, int]

    def __init__(self, adj_view_def: AdjustedViewDef, view_col_numbers: dict[str, int]):
        """view_col_numbers must contain both predictive and adjusting views."""
        self.__adj_view_def = adj_view_def
        predictive_view_sizes = {}
        for view_name in adj_view_def.predictive_view_names_seq():
            predictive_view_sizes[view_name] = view_col_numbers[view_name]
        self.__predictive_views_to_concat = MvToConcatMapper(view_sizes=predictive_view_sizes)
        self.__predictive_features_num = self.__predictive_views_to_concat.concat_size()
        self.__view_col_numbers = view_col_numbers

    def _predictive_views_to_concat(self) -> MvToConcatMapper:
        return self.__predictive_views_to_concat

    def _adj_view_def(self) -> AdjustedViewDef:
        return self.__adj_view_def

    def predictive_features_mask_len_mv(self) -> int:
        return self.__predictive_features_num

    def add_adjusting_views(self, current_selection: dict[str, BoolListLike]) -> dict[str, SparseBoolList]:
        """Returned dict is sorted by keys. Views with 0 active features are still included in the result.
        Adjusting views that adjust predictive views with zero features are included, but with zero features,
        unless that adjusting view is specified as having selected features in the current_selection."""
        return add_adjusting_views(
            current_selection=current_selection, adj_view_def=self.__adj_view_def, view_sizes=self.__view_col_numbers)

    def __str__(self) -> str:
        res = ""
        res += "Number of predictive features: " + str(self.__predictive_features_num) + "\n"
        res += "Adjusted view definitions:\n" + str(self.__adj_view_def) + "\n"
        res += "Predictive views to concatenated:\n" + str(self.__predictive_views_to_concat) + "\n"
        res += "View column numbers: " + str(self.__view_col_numbers) + "\n"
        return res

    def contextualize_all(
            self, pop: Iterable[ConfidentPredictiveIndividualMV]) -> Sequence[PeculiarIndividualWithContextMV]:
        from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
        from individual.peculiar_individual_with_context_mv import PeculiarIndividualWithContextMV
        from individual.confident_predictive_individual import ConfidentPredictiveIndividualSparseMV

        adj_view_def = self.__adj_view_def.make_all_views_predictive()
        hp_manager = MaskMvHpManager(adj_view_def=adj_view_def, view_col_numbers=self.__view_col_numbers)
        res = []
        for individual in pop:
            to_contextualize = ConfidentPredictiveIndividualSparseMV.create_from_individual(
                individual=individual, hp_manager=self)
            res.append(PeculiarIndividualWithContextMV(individual=to_contextualize, hp_manager=hp_manager))
        return res


