from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterable

from hyperparam_manager.hyperparam_manager import HyperparamManager
from util.list_like import ListLike, BoolListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from util.feature_space_lifter import FeatureSpaceLifterMV
    from input_data.evaluation_ready_input_data import EvaluationReadyInputData
    from individual.confident_predictive_individual import ConfidentPredictiveIndividualMV
    from individual.peculiar_individual_with_context_mv import PeculiarIndividualWithContextMV


class MvHyperparamManager(HyperparamManager,ABC):
    """Multi-view hyperparameter manager. Can filter the views separately. Considers the difference between
    predictive (directly used for predicting) and used (predictive and adjusting) features."""

    @abstractmethod
    def n_predictive_features(self, hyperparams: ListLike) -> int:
        """Including predictive but not adjusting.
        Refers to the predictive features used by these specific hyperparams."""
        raise NotImplementedError()

    def n_used_features(self, hyperparams: ListLike) -> int:
        """Including both predictive and adjusting."""
        return sum(v.sum() for v in self.used_feature_masks(hyperparams=hyperparams).values())

    @abstractmethod
    def used_feature_masks(self, hyperparams: ListLike) -> dict[str, BoolListLike]:
        """hyperparams is an assignment of the hyperparameters.
        Returned dict is sorted by keys. Includes both predictive and adjusting features.
        Views with 0 active features are still included in the result.
        Adjusting views that adjust predictive views with zero features are included, but with zero features,
        unless that adjusting view is specified as having selected features in the current_selection."""
        raise NotImplementedError()

    def collapsed_used_features_mask(self, hyperparams: ListLike) -> BoolListLike:
        """Obtained by concatenating all the views together (both predictive and adjusting views) in alphabetical order.
        Returned mask is in the form of a Boolean list.
        This is used also to create masked predictors, where input is collapsed and then masked."""
        res = SparseBoolListBySet()
        for v in self.used_feature_masks(hyperparams=hyperparams).values():
            res.extend(v)
        return res

    @abstractmethod
    def predictive_features_mask_len_mv(self) -> int:
        """Since MV HP managers do not need a passed hyperparam instance to return the length of the mask."""
        raise NotImplementedError()

    def predictive_features_mask_len(self, hyperparams: ListLike) -> int:
        return self.predictive_features_mask_len_mv()

    def feature_space_lifter(self, hyperparams: ListLike) -> FeatureSpaceLifterMV:
        from util.feature_space_lifter import FeatureSpaceLifter, FeatureSpaceLifterMV
        masks = self.used_feature_masks(hyperparams=hyperparams)
        single_view_lifters = {
            view_name: FeatureSpaceLifter(active_features=single_view_mask)
            for view_name, single_view_mask in masks.items()}
        return FeatureSpaceLifterMV(single_view_lifters=single_view_lifters)

    def filter_evaluation_ready_data(
            self, hyperparams: ListLike, data: EvaluationReadyInputData) -> EvaluationReadyInputData:
        """Filters the features according to the passed hyperparams."""
        masks = self.used_feature_masks(hyperparams=hyperparams)
        return data.select_features(masks=masks)

    @abstractmethod
    def  contextualize_all(
            self, pop: Iterable[ConfidentPredictiveIndividualMV]) -> Sequence[PeculiarIndividualWithContextMV]:
        """Returns a contextualized version of the individuals."""
        raise NotImplementedError()