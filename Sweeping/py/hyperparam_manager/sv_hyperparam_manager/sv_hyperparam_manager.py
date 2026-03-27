from __future__ import annotations
from abc import ABC, abstractmethod

from hyperparam_manager.hyperparam_manager import HyperparamManager
from util.feature_space_lifter import FeatureSpaceLifter
from util.list_like import ListLike


class SvHyperparamManager(HyperparamManager, ABC):

    def n_used_features(self, hyperparams: ListLike) -> int:
        """Including both predictive and adjusting. Since this kind of HP manager does not distinguish
        between views it cannot consider adjusting views, so every feature is considered predictive and used."""
        return self.n_predictive_features(hyperparams=hyperparams)

    @abstractmethod
    def downlift(self, lifter: FeatureSpaceLifter) -> SvHyperparamManager:
        """Used to downlift for example from the set of active features to the set of all features."""
        raise NotImplementedError()
